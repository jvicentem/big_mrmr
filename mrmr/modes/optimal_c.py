from collections import deque
from datetime import datetime
import logging
import time
import statistics

from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

from modes.abstract_mode import AbstractMode

class OptimalC(AbstractMode):    
    def __init__(self, df, replace_na, target, k, top_best_solutions, must_included_vars, max_mins, df_count):
        super().__init__(df, replace_na, target, k, top_best_solutions, must_included_vars, max_mins, df_count)
        
        self.mrmr_scores = {}
        self.cols_processed = [] 

        self.mrmr_best_partial_score = [-9999] * self.k  

    def calculate_optimal_vars(self):
        self.remove_high_card_vars()
        
        self.start_time = time.time()

        logger = logging.getLogger('optimal_c')

        def _iterate(a, cols):
            if a != self.target:
                x_mis = []
                target_mi = None
                processed_cols = []

                comb_target = self.comb(a, self.target)
                
                target_mi = self.mi_cache[comb_target]

                processed_cols.append(comb_target)

                # check if it's worth it to continue...
                # if in the best possible scenario the mrmr score is better than the score in kth position...
                possible_good_mi = target_mi - statistics.mean([0] * (len(self.df.columns)-2))

                if possible_good_mi >= self.mrmr_best_partial_score[self.k - 1]:           
                    worth_continue = True

                    for b in cols:
                        if a != b and b != self.target:
                            comb = self.comb(a, b)
                            self.mi_cache[comb] = self.mi(a, b)
                            x_mis.append(self.mi_cache[comb])
                            processed_cols.append(comb)

                            # check if it's worth it to continue...  
                            aux_x_mis = x_mis.copy()

                            mi_cache_keys = list(self.mi_cache.keys())
                            
                            for c in mi_cache_keys:
                                if c not in processed_cols and a in c:                                        
                                    aux_x_mis.append(self.mi_cache[c])
                                    
                            aux_x_mis = aux_x_mis + [0]*(len(self.df.columns)-len(aux_x_mis)-2)                    

                            possible_good_mi = target_mi - statistics.mean(aux_x_mis)
                            worth_continue = possible_good_mi >= self.mrmr_best_partial_score[self.k - 1]

                            if not worth_continue:       
                                self.cols_processed.append(a)    
    
                            if (time.time() - self.start_time) / 60 > self.max_mins:
                                self.cols_processed.append(a)
                                self.mrmr_scores[a] = None 
                                return None

                    if worth_continue:
                        final_mi = target_mi - statistics.mean(x_mis)

                        if final_mi >= self.mrmr_best_partial_score[self.k - 1]:
                            self.mrmr_best_partial_score[self.k - 1] = final_mi
                            self.mrmr_best_partial_score.sort(reverse=True)
                            self.mrmr_best_partial_score = self.mrmr_best_partial_score[:self.k]
                            self.mrmr_scores[a] = [final_mi]
                    
                        self.cols_processed.append(a)
                
                        return final_mi
                    else:  
                        self.cols_processed.append(a)
                else:                         
                    self.cols_processed.append(a)
        
        def _calc_target_mi(col):
            if col != self.target:
                comb = self.comb(col, self.target)
                self.mi_cache[comb] = self.mi(col, self.target)    
                return (col, [self.mi_cache[comb]])
            else:
                return (col, [None])
        
        logger.info('Calculating target mis...')
        logger.info(datetime.now())
    
        self.ent_cache[self.target] = self.ent([self.target])

        target_mis = Parallel(n_jobs=-1, require='sharedmem')(delayed(_calc_target_mi)(col) for col in tqdm(self.df.columns))
    
        target_mis = pd.DataFrame(dict(target_mis)).sort_values(axis=1, by=0, ascending=False)
    
        cols_to_process = []
        aux_col = deque(list(target_mis.columns))
    
        for _ in list(target_mis.columns):
            aux_col.rotate(-1)
            cols_to_process.append(list(aux_col))
        
        logger.info('Calculating mrmr...')
        logger.info(datetime.now())
                
        _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(_iterate)(col, cols_to_process[ix]) 
                                                                for ix, col in tqdm(
                                                                    (x for x in list(enumerate(list(target_mis.columns))) if (time.time() - self.start_time) / 60 <= self.max_mins)
                                                                    )
                                                    )

        return (pd.DataFrame(self.mrmr_scores)
                  .sort_values(axis=1, by=0, ascending=False)
                  .iloc[:, :self.k]
                  .rename(columns={0: 'features_set', 1: 'mrmr'})
                  )
