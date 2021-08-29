from datetime import datetime
import itertools
import math
import time
import statistics

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.cluster._supervised import _generalized_average

from cython_modules._expected_mutual_info_fast import expected_mutual_information
from modes.abstract_mode import AbstractMode

class OptimalK(AbstractMode):    
    def __init__(self, df, replace_na, target, k, top_best_solutions, must_included_vars, max_mins, df_count):
        super().__init__(df, replace_na, target, k, top_best_solutions, must_included_vars, max_mins, df_count)

        self.group_residual_cats = True
        self.cancelled_partial_solutions = []  

        self.mrmr_best_partial_score = [('', -9999)] * top_best_solutions    

    def calculate_optimal_vars(self):
        self.start_time = time.time()

        # 0. Take into account that user can configurate variables that must be in the solution.            
        # 1. Get all combinations of k features    

        self._remove_high_card_vars()

        print('Calculating target adj mis...')
        print(datetime.now())                   

        target_mis = Parallel(n_jobs=-1, require='sharedmem')(delayed(self._calc_target_adj_mi)(col) for col in tqdm(self.df.columns))                                

        target_mis = pd.DataFrame(dict(target_mis)).sort_values(axis=1, by=0, ascending=False)          
        target_mis = list(target_mis.columns)
        target_mis.remove(self.target)

        combs = itertools.combinations(target_mis, self.k)

        # 2. Iterate through them
        # 2.2. Inside each solution, calculate the MRMR for each variable.                    

        def _iterate_sols(c, first_it):
            if (time.time() - self.start_time) / 60 <= self.max_mins:
                in_cancelled_sols = any([set(c[0:i]) == set(c[0:i]).intersection(cs) for cs in self.cancelled_partial_solutions for i in range(2, self.k)])
                must_vars_in_c = not self.must_included_vars or all(col in c for col in self.must_included_vars)

                if not in_cancelled_sols and must_vars_in_c:
                    partial_mrmr = []

                    worth_continue = True

                    for ix, col in enumerate(c):
                        x_mean_mi = statistics.mean([self._adj_mrmr_cache(col, x) for x in c if x != col])
                        col_mrmr = self._adj_mrmr_cache(col, self.target) - x_mean_mi
                        
                        best_possible_mrmr = partial_mrmr + [col_mrmr]
                        best_possible_mrmr = statistics.mean( best_possible_mrmr + ([1.0] * (self.k - len(best_possible_mrmr))) )

                        if not first_it:  
                            worth_continue = best_possible_mrmr >= self.mrmr_best_partial_score[self.top_best_solutions - 1][1]
                            
                            if worth_continue:
                                partial_mrmr.append(col_mrmr) 
                            else:
                                self.cancelled_partial_solutions.append(set(c[0:(ix+1)]))
                                break
                        else:
                            partial_mrmr.append(col_mrmr) 

                    if worth_continue:
                        self.mrmr_best_partial_score.append( (c, statistics.mean(partial_mrmr)) )
                        self.mrmr_best_partial_score = sorted(self.mrmr_best_partial_score, key=lambda x: x[1], reverse=True)
                        self.mrmr_best_partial_score = self.mrmr_best_partial_score[0:(self.top_best_solutions)]                        
        
        print('Calculating mrmr...')

        n_starting_coms = (math.factorial(len(target_mis)) / 
                            (math.factorial((len(target_mis) - self.k)) * math.factorial(self.k))
                            )

        _ = _iterate_sols(next(combs), True)

        tqdm._instances.clear()
        _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(_iterate_sols)(c, False) for c in tqdm(combs, total=n_starting_coms))                   

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

    def _adj_mrmr_cache(self, a, b):
        return self.adjusted_mutual_info_score(a, b)    

    def _remove_high_card_vars(self, thresh = 0.5):        
        for k in self.df.columns:
            if k != self.target:
                self.label_count_cache[k] = self.count_cats([k], return_counts=True)

                n_unique_values = len(self.label_count_cache[k])

                if n_unique_values/self.df_count >= thresh:                    
                    del self.label_count_cache[k]
                    self.df = self.df.drop(k)
                    print(f'Variable "{k}" removed because high cardinality (ratio > {thresh} of whole dataset size). If it\'s numerical, consider including it in the cont_vars parameter or remove it.')                    

    def adjusted_mutual_info_score(self, a, b):
        comb = self.comb(a, b)

        if comb in self.ami_cache:
            return self.ami_cache[comb]
        else:           
            # label_count = self.count_cats([a, b], return_counts=True)
            
            # classes = label_count[a].unique()

            # clusters = label_count[b].unique()
            self.label_count_cache[comb] = self.count_cats([a, b], return_counts=True)
            
            classes = self.label_count_cache[comb][a].unique()

            clusters = self.label_count_cache[comb][b].unique()

            # Special limit cases: no clustering since the data is not split.
            # This is a perfect match hence return 1.0.
            if (classes.shape[0] == clusters.shape[0] == 1 or classes.shape[0] == clusters.shape[0] == 0):
                return 1.0        

            # mi = self.mi(a, b)        

            # aux_count_a = label_count[[a]].copy()
            # aux_count_a['aux'] = 1

            # aux_count_b = label_count[[b]].copy()
            # aux_count_b['aux'] = 1    
            self.mi_cache[comb] = self.mi(a, b)        

            aux_count_a = self.label_count_cache[comb][[a, 'count']]#.copy()
            # aux_count_a['aux'] = 1

            aux_count_b = self.label_count_cache[comb][[b, 'count']]#.copy()
            # aux_count_b['aux'] = 1            
                        
            emi = expected_mutual_information(aux_count_a.groupby(a).sum()['count'].values.astype('int32'), 
                                              aux_count_b.groupby(b).sum()['count'].values.astype('int32'), 
                                              self.df_count,
                                              len(classes), 
                                              len(clusters)
                                              )

            # emi_py = expected_mutual_information_py(aux_count_a.groupby(a).count()['aux'].values.astype('int32'), 
            #                                         aux_count_b.groupby(b).count()['aux'].values.astype('int32'), 
            #                                         self.df_count,
            #                                         len(classes), 
            #                                         len(clusters)
            #                                         )

            # ent_a = self.ent([a])
            # ent_b = self.ent([b])
            # h_true, h_pred = ent_a, ent_b
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
            # return (mi - emi) / denominator    
            return (self.mi_cache[comb] - emi) / denominator    