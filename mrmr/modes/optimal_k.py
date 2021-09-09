from datetime import datetime
import itertools
import logging
import math
import random
import statistics
import sys
import time

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.cluster._supervised import _generalized_average

from cython_modules._expected_mutual_info_fast import expected_mutual_information
from modes.abstract_mode import AbstractMode

class OptimalK(AbstractMode):    
    def __init__(self, df, replace_na, target, k, top_best_solutions, must_included_vars, max_mins, df_count, seed):
        super().__init__(df, replace_na, target, k, top_best_solutions, must_included_vars, max_mins, df_count)

        self.group_residual_cats = True
        self.cancelled_partial_solutions = []  
        self.seed = seed

        self.mrmr_best_partial_score = [('', -9999)] * top_best_solutions 

    def calculate_optimal_vars(self):
        self.remove_high_card_vars()
        
        self.start_time = time.time()

        # 0. Take into account that user can configurate variables that must be in the solution.            
        # 1. Get all combinations of k features            

        logger = logging.getLogger('optimal_k')

        logger.info('Calculating target adj mis...')
        logger.info(datetime.now())                   

        target_mis = Parallel(n_jobs=-1, require='sharedmem')(delayed(self._calc_target_adj_mi)(col) for col in tqdm(self.df.columns))                                

        target_mis = pd.DataFrame(dict(target_mis)).sort_values(axis=1, by=0, ascending=False)          
        target_mis = list(target_mis.columns)     
        target_mis.remove(self.target)

        n_starting_coms = (math.factorial(len(target_mis)) / 
                           (math.factorial((len(target_mis) - self.k)) * math.factorial(self.k))
                          )        
        
        if self.max_mins:
            random.seed(self.seed)
            sample = random.sample
            combs = target_mis[:self.k]
        else:
            combs = itertools.combinations(target_mis, self.k)

        # 2. Iterate through them
        # 2.2. Inside each solution, calculate the MRMR for each variable.                    

        def _iterate_sols(c, first_it):
            in_cancelled_sols = any([set(c[0:i]) == set(cs) for cs in self.cancelled_partial_solutions for i in range(2, self.k)])
            must_vars_in_c = not self.must_included_vars or all(col in c for col in self.must_included_vars)                

            if not in_cancelled_sols and must_vars_in_c:
                partial_mrmr = []

                worth_continue = True

                for ix, col in enumerate(c):
                    partial_col_mrmr = []

                    for x in c:
                        if x != col:
                            if len(partial_col_mrmr) == 0:
                                partial_col_mrmr.append(self._adj_mi_cache(col, x))
                            else:             
                                if not first_it:
                                    partial_x_mean_mi = statistics.mean(partial_col_mrmr + ([0.0] * (self.k-1 - len(partial_col_mrmr))))

                                    partial_val_col_mrmr = self._adj_mi_cache(col, self.target) - partial_x_mean_mi

                                    best_partial_possible_mrmr = partial_mrmr + [partial_val_col_mrmr]
                                    best_partial_possible_mrmr = best_partial_possible_mrmr + [self._adj_mi_cache(aux_c, self.target) for aux_c in c[ix:len(c)]] #adj_mrmr(aux_c, target) - 0
                                    best_partial_possible_mrmr = statistics.mean(best_partial_possible_mrmr)

                                    worth_continue = best_partial_possible_mrmr > self.mrmr_best_partial_score[self.top_best_solutions - 1][1]

                                    if not worth_continue:
                                        if len(c[0:(ix+1)]) < self.k:
                                            self.cancelled_partial_solutions.append(set(c[0:(ix+1)]))
                                        break       
                                    else:
                                        partial_col_mrmr.append(self._adj_mi_cache(col, x))       
                                else:
                                    partial_col_mrmr.append(self._adj_mi_cache(col, x))                              
                    
                    if not first_it:                              
                        if worth_continue:
                            col_mrmr = self._adj_mi_cache(col, self.target) - statistics.mean(partial_col_mrmr)

                            best_possible_mrmr = partial_mrmr + [col_mrmr]
                            best_possible_mrmr = statistics.mean( best_possible_mrmr + ([1.0] * (self.k - len(best_possible_mrmr))) )
                            worth_continue = best_possible_mrmr > self.mrmr_best_partial_score[self.top_best_solutions - 1][1]

                            if worth_continue:
                                partial_mrmr.append(col_mrmr)
                            else:
                                if len(c[0:(ix+1)]) < self.k:
                                    self.cancelled_partial_solutions.append(set(c[0:(ix+1)]))
                                break                                
                        else:
                            break
                    else:
                        col_mrmr = self._adj_mi_cache(col, self.target) - statistics.mean(partial_col_mrmr)
                        partial_mrmr.append(col_mrmr) 

                if worth_continue:
                    self.mrmr_best_partial_score.append( (c, statistics.mean(partial_mrmr)) )
                    self.mrmr_best_partial_score = sorted(self.mrmr_best_partial_score, key=lambda x: x[1], reverse=True)
                    self.mrmr_best_partial_score = self.mrmr_best_partial_score[0:(self.top_best_solutions)]                        
        
        logger.info('Calculating mrmr...')
        logger.info(datetime.now())

        if self.max_mins:
            _ = _iterate_sols(combs, True)
        else:            
            _ = _iterate_sols(next(combs), True)
 
        tqdm._instances.clear()        

        if self.max_mins:
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(_iterate_sols)(c, False) for c in tqdm( (sample(target_mis, self.k) for x in range(0, sys.maxsize) if (time.time() - self.start_time) / 60 <= self.max_mins)  ))           
        else:
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(_iterate_sols)(c, False) for c in tqdm((x for x in combs if (time.time() - self.start_time) / 60 <= self.max_mins), total=n_starting_coms))                   

        # 2.2.1. When the mean MRMR of some of the variables cannot reach the mean MRMR of other
        # solution, we stop iterating through the variables of that solution, and that partial
        # solution is marked as not worth to explore (if future solutions containing that subsoltion
        # come).
        # 2.2.1.1. Use adjusted mutual information (by the moment, only for this solution).
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
        # 2.2.2. If a solution has a better mean MRMR than the previous one, it is chosen as the 
        # solution at the moment and it's mean MRMR is the score to beat for other solutions.

        return (pd.DataFrame(self.mrmr_best_partial_score)
                  .sort_values(by=1, ascending=False)
                  .iloc[:, :self.top_best_solutions]
                  .rename(columns={0: 'features_set', 1: 'adjusted_mrmr'})
                  )
        
    def _calc_target_adj_mi(self, col):                
        if col != self.target:
            comb = self.comb(col, self.target)
            self.ami_cache[comb] = self.adjusted_mutual_info_score(col, self.target)
            return (col, [self.ami_cache[comb]])
        else:
            return (col, [None])
    
    def _adj_mi_cache(self, a, b):
        if a != b:
            comb = self.comb(a, b)
            self.ami_cache[comb] = self.adjusted_mutual_info_score(a, b)
            return self.ami_cache[comb]
        else:
            return None           

    def adjusted_mutual_info_score(self, a, b):
        comb = self.comb(a, b)

        if comb in self.ami_cache:
            return self.ami_cache[comb]
        else:           
            self.label_count_cache[comb] = self.count_cats([a, b], return_counts=True)
            
            classes = self.label_count_cache[comb][a].unique()

            clusters = self.label_count_cache[comb][b].unique()

            # Special limit cases: no clustering since the data is not split.
            # This is a perfect match hence return 1.0.
            if (classes.shape[0] == clusters.shape[0] == 1 or classes.shape[0] == clusters.shape[0] == 0):
                return 1.0        

            self.mi_cache[comb] = self.mi(a, b)        

            aux_count_a = self.label_count_cache[comb][[a, 'count']]

            aux_count_b = self.label_count_cache[comb][[b, 'count']]    
                        
            emi = expected_mutual_information(aux_count_a.groupby(a).sum()['count'].values.astype('int32'), 
                                              aux_count_b.groupby(b).sum()['count'].values.astype('int32'), 
                                              self.df_count,
                                              len(classes), 
                                              len(clusters)
                                              )

            self.ent_cache[a] = self.ent([a])
            self.ent_cache[b] = self.ent([b])
            h_true, h_pred = self.ent_cache[a], self.ent_cache[b]            
            
            normalizer = _generalized_average(h_true, h_pred, 'arithmetic')

            denominator = normalizer - emi

            # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
            # normalizer should always be >= emi, but because of floating-point
            # representation, sometimes emi is slightly larger. Correct this
            # by preserving the sign.
            if denominator < 0:
                denominator = min(denominator, -np.finfo('float64').eps)
            else:
                denominator = max(denominator, np.finfo('float64').eps)
            
            return (self.mi_cache[comb] - emi) / denominator    