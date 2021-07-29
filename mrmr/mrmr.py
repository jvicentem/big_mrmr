from collections import deque
from datetime import datetime
from functools import lru_cache
import itertools
import math
import statistics
from sys import maxsize
import time

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import pandas as pd
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from scipy.sparse import csr_matrix
from scipy.stats import entropy as sc_entropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information   
from sklearn.metrics.cluster._supervised import _generalized_average

from _expected_mutual_info_fast import expected_mutual_information, prueba
from expected_mutual_info_fast import expected_mutual_information_py


class MRMR:
    def __init__(self, df, target, k=10, subset=[], cont_vars=[], replace_na=False, 
                 optimal_k=False, top_best_solutions=5, must_included_vars=[], 
                 max_mins=10):
        
        self.subset = df.columns if not subset else subset 

        if target not in self.subset:
            self.subset = self.subset + [target]

        self.df = df.select(*self.subset)
        self.replace_na = replace_na
        self.target = target
        self.k = k
        self.mrmr_scores = {}
        self.mrmr_best_partial_score = [-9999] * self.k      
        self.optimal_k = optimal_k
        self.top_best_solutions = top_best_solutions
        self.must_included_vars = must_included_vars
        
        self.label_count_cache = {}
        self.ent_cache = {}
        self.mi_cache = {}
        self.ami_cache = {}
        self.mrmr_debug = {}
        self.start_time = None
        self.max_mins = max_mins
        self.cols_processed = []      

        self.cancelled_partial_solutions = []  

        self.group_residual_cats = False
        
        for v in self.subset:
            if v in cont_vars:
                self.df = (self.df
                               .withColumn(v, 
                                           f.ntile(10).over(Window.partitionBy(f.lit(1)).orderBy(v)))
                          )
                
                if self.replace_na:
                    self.df = self.df.na.fill(-1.0, subset=[v])                
            else:
                self.df = self.df.withColumn(v, f.col(v).cast('string'))
                
                if self.replace_na:
                    self.df = self.df.na.fill('Null', subset=[v])
                
        if not self.optimal_k:
            self.df = self.df.cache()

        self.df_count = self.df.count()
        
    def comb(self, a, b):
        if a < b:
            return f'{a}-{b}'
        else:
            return f'{b}-{a}'
    
    def ent(self, features):
        if len(features) == 1:
            a = features[0]
            
            if a in self.ent_cache:
                return self.ent_cache[a]
            else:
                self.label_count_cache[a] = self.count_cats([a])                            
                return sc_entropy(self.label_count_cache[a]['ratio'], base=2)
        else:
            comb = self.comb(features[0], features[1])
            
            if comb in self.ent_cache:
                return self.ent_cache[comb]  
            else:                    
                self.label_count_cache[comb] = self.count_cats(features)                
                return sc_entropy(self.label_count_cache[comb]['ratio'], base=2)
    
    def mi(self, a, b):
        self.ent_cache[a] = self.ent([a])
        
        # if a in self.label_count_cache:
        #     del self.label_count_cache[a]
            
        self.ent_cache[b] = self.ent([b])
        
        # if b in self.label_count_cache:
        #     del self.label_count_cache[b]
            
        comb = self.comb(a, b)
        self.ent_cache[comb] = self.ent([a, b])

        # if comb in self.label_count_cache:
        #     del self.label_count_cache[comb]
        
        return self.ent_cache[a] + self.ent_cache[b] - self.ent_cache[comb]
        
    def count_cats(self, features, return_counts=False):
        # comb = None
        if len(features) == 1:
            a = features[0]
            
            if a in self.label_count_cache: # UNCOMMENT THIS
                return self.label_count_cache[a]
        else:
            comb = self.comb(features[0], features[1])
            if comb in self.label_count_cache: # and not conf_mat: # UNCOMMENT THIS
                return self.label_count_cache[comb]
            # elif comb in self.label_count_cache and conf_mat and comb in self.conf_mat_cache:
            #     return self.label_count_cache[comb], self.conf_mat_cache[comb]

        ratio_df = (self.df
                    .select(*features)
                    .groupBy(*features)
                    .count()
                    .withColumn('ratio', 
                                f.col('count') / f.lit(self.df_count)
                                )                   
                    )       

        if self.group_residual_cats:
            residual_df = ratio_df.filter(f.col('ratio') <= 0.01)

            for fe in features:
                residual_df = residual_df.withColumn(fe, f.lit('OTHER'))
            
            residual_df = (residual_df
                            .groupBy(*features)
                            .agg(f.sum('ratio').alias('ratio'),
                                 f.sum('count').alias('count'))
                           )

            ratio_df = ratio_df.filter(f.col('ratio') > 0.01).union(residual_df.select(ratio_df.columns))

        # if comb in self.label_count_cache:
        #     ratio_pdf = self.label_count_cache[comb]

        #     if conf_mat and 'count' not in ratio_pdf.columns:
        #         ratio_df = ratio_df.cache() 
        #         ratio_pdf = ratio_df.toPandas()

        #         ratio_pdf['ratio'] = ratio_pdf['ratio'].astype('float32')
        # elif not conf_mat and comb not in self.label_count_cache:

        if not return_counts:
            ratio_pdf = (ratio_df
                            .drop('count')
                            .select('ratio')
                            .toPandas()              
                        )             
        else:
            ratio_pdf = ratio_df.toPandas()         
            ratio_pdf['count'] = ratio_pdf['count'].astype('int32')    

        ratio_pdf['ratio'] = ratio_pdf['ratio'].astype('float32') 
         
        # else:
        #     ratio_df = ratio_df.cache() 
        #     ratio_pdf = ratio_df.toPandas()
        #     ratio_pdf['ratio'] = ratio_pdf['ratio'].astype('float32')                  
                    
        # if conf_mat and len(features) > 1:
        #     feat_a_vals = ratio_pdf[features[0]].unique().tolist()

        #     if len(feat_a_vals) * len(ratio_pdf[features[1]].unique().tolist()) > 10000000:
        #         print('''WARNING: This is gonna create a large confusion matrix. 
        #                  It\'s Very likely it will freeze your driver/local machine...''')

        #     conf_mat = csr_matrix(ratio_df
        #                             .groupBy(*features)
        #                             .pivot(features[0], feat_a_vals)
        #                             .agg(f.first('count'))
        #                             .fillna(0)
        #                             .groupBy(features[1])
        #                             .agg([f.sum(str(x)).alias(str(x)) for x in feat_a_vals])
        #                             .drop(*features)
        #                             .toPandas()
        #                             .to_numpy()
        #                           ) 
            
        #     if conf_mat and 'count' not in ratio_pdf.columns:
        #         ratio_df = ratio_df.unpersist()

        #     return ratio_pdf, conf_mat                                 
        # else:
        return ratio_pdf
    
    def mrmr(self):       
        self.start_time = time.time()
        print('MRMR calculation starting...')
        print(datetime.now())
        
        if not self.optimal_k:
            def _iterate(a, cols):
                if (time.time() - self.start_time) / 60 <= self.max_mins:
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
                                
                                        return -9997
            
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
                        
                                return -9998
                        else:                         
                            self.cols_processed.append(a)
                                
                            return -9999
                else:
                    self.cols_processed.append(a)
                        
                    self.mrmr_scores[a] = [None]
                    
                    return None
            
            def _calc_target_mi(col):
                if col != self.target:
                    comb = self.comb(col, self.target)
                    self.mi_cache[comb] = self.mi(col, self.target)    
                    return (col, [self.mi_cache[comb]])
                else:
                    return (col, [None])
            
            print('Calculating target mis...')
            print(datetime.now())
        
            self.ent_cache[self.target] = self.ent([self.target])

            # target_mis = []
            # for col in tqdm(self.df.columns):
            #     target_mis.append(_calc_target_mi(col))

            target_mis = Parallel(n_jobs=-1, require='sharedmem')(delayed(_calc_target_mi)(col) for col in tqdm(self.df.columns))
        
            target_mis = pd.DataFrame(dict(target_mis)).sort_values(axis=1, by=0, ascending=False)
        
            cols_to_process = []
            aux_col = deque(list(target_mis.columns))
        
            for c in list(target_mis.columns):
                aux_col.rotate(-1)
                cols_to_process.append(list(aux_col))
            
            print('Calculating mrmr...')
            print(datetime.now())

            # debug_mrmr = []
            # for ix, col in tqdm(list(enumerate(list(target_mis.columns)))):
            #     debug_mrmr.append(_iterate(col, cols_to_process[ix]))
                    
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(_iterate)(col, cols_to_process[ix]) 
                                                                  for ix, col in tqdm(list(enumerate(list(target_mis.columns)))))

            return pd.DataFrame(self.mrmr_scores).sort_values(axis=1, by=0, ascending=False).iloc[:, :self.k]
        else: 
            # 0. Take into account that user can configurate variables that must be in the solution.            
            # 1. Get all combinations of k features
            
            new_mrmr_best_partial_score = []
            for _ in range(self.top_best_solutions):
                new_mrmr_best_partial_score.append(('', -9999))
            self.mrmr_best_partial_score = new_mrmr_best_partial_score

            thresh = 0.5
            for k in self.df.columns:
                if k != self.target:
                    self.label_count_cache[k] = self.count_cats([k], return_counts=True)

                    n_unique_values = len(self.label_count_cache[k])

                    if n_unique_values/self.df_count >= thresh:                    
                        del self.label_count_cache[k]
                        self.df = self.df.drop(k)
                        print(f'Variable "{k}" removed because high cardinality (ratio > {thresh} of whole dataset size). If it\'s numerical, consider including it in the cont_vars parameter or remove it.')

            def _calc_target_adj_mi(col):                
                if col != self.target:
                    comb = self.comb(col, self.target)
                    self.ami_cache[comb] = self.adjusted_mutual_info_score(col, self.target)
                    return (col, [self.ami_cache[comb]])
                else:
                    return (col, [None])

            print('Calculating target adj mis...')
            print(datetime.now())  

            self.group_residual_cats = True            

            target_mis = Parallel(n_jobs=-1, require='sharedmem')(delayed(_calc_target_adj_mi)(col) for col in tqdm(self.df.columns))                    
            
            # target_mis = []
            # for col in tqdm(self.df.columns):
            #     target_mis.append(_calc_target_adj_mi(col))

            target_mis = pd.DataFrame(dict(target_mis)).sort_values(axis=1, by=0, ascending=False)          
            target_mis = list(target_mis.columns)
            target_mis.remove(self.target)

            combs = itertools.combinations(target_mis, self.k)

            # 2. Iterate through them
            # 2.2. Inside each solution, calculate the MRMR for each variable.            

            def adj_mrmr_cache(a, b):
                # comb = self.comb(a, b)

                return self.adjusted_mutual_info_score(a, b)
                # if comb not in self.ami_cache:
                #     self.ami_cache[comb] = self.adjusted_mutual_info_score(a, b)        

                # return self.ami_cache[comb]                

            def _iterate_sols(c, first_it):
                in_cancelled_sols = any([c[0:i] in self.cancelled_partial_solutions for i in range(2, self.k)])
                must_vars_in_c = not self.must_included_vars or all(col in c for col in self.must_included_vars)

                if not in_cancelled_sols and must_vars_in_c:
                    partial_mrmr = []

                    worth_continue = True

                    for ix, col in enumerate(c):
                        x_mean_mi = statistics.mean([adj_mrmr_cache(col, x) for x in c if x != col])
                        col_mrmr = adj_mrmr_cache(col, self.target) - x_mean_mi
                        # col_mrmr = self.ami_cache[self.comb(col, self.target)] - x_mean_mi
                        
                        best_possible_mrmr = partial_mrmr + [col_mrmr]
                        best_possible_mrmr = statistics.mean( best_possible_mrmr + ([1.0] * (self.k - len(best_possible_mrmr))) )

                        if not first_it:  
                            worth_continue = best_possible_mrmr >= self.mrmr_best_partial_score[self.top_best_solutions - 1][1]
                            
                            if worth_continue:
                                partial_mrmr.append(col_mrmr) 
                            else:
                                self.cancelled_partial_solutions.append(c[0:(ix+1)])
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
            
            # t = tqdm(combs, total=n_starting_coms)
            # for c in t:
            #     t.set_description(str(c))
            #     _iterate_sols(c, False)
            
            print(datetime.now())            

            # 2.2.1. When the mean MRMR of some of the variables cannot reach the mean MRMR of other
            # solution, we stop iterating through the variables of that solution, and that partial
            # solution is marked as not worth to explore (if future solutions containing that subsoltion
            # come).
            # 2.2.1.1. Use adjusted mutual information (by the moment, only for this solution).
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
            # 2.2.2. If a solution has a better mean MRMR than the previous one, it is chosen as the 
            # solution at the moment and it's mean MRMR is the score to beat for other solutions.

            return pd.DataFrame(self.mrmr_best_partial_score).sort_values(by=1, ascending=False).iloc[:, :self.top_best_solutions], None

    @lru_cache(maxsize=1024)
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
