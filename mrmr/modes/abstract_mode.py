from abc import abstractmethod

from pyspark.sql import functions as f
from scipy.stats import entropy as sc_entropy

class AbstractMode:
    def __init__(self, df, replace_na, target, k, top_best_solutions, must_included_vars, max_mins, df_count):
        self.df = df
        self.label_count_cache = {}
        self.ent_cache = {}
        self.mi_cache = {}
        self.ami_cache = {}
        self.replace_na = replace_na
        self.target = target
        self.k = k                
        self.top_best_solutions = top_best_solutions
        self.must_included_vars = must_included_vars

        self.mrmr_best_partial_score = []
        
        self.max_mins = max_mins
        self.start_time = None
        self.group_residual_cats = False

        self.df_count = df_count

    @abstractmethod
    def calculate_optimal_vars(self):
        pass

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
                return sc_entropy(self.label_count_cache[a]['ratio'])
        else:
            comb = self.comb(features[0], features[1])
            
            if comb in self.ent_cache:
                return self.ent_cache[comb]  
            else:                    
                self.label_count_cache[comb] = self.count_cats(features)                
                return sc_entropy(self.label_count_cache[comb]['ratio'])
    
    def mi(self, a, b):
        self.ent_cache[a] = self.ent([a])
        
        self.ent_cache[b] = self.ent([b])    
            
        comb = self.comb(a, b)
        self.ent_cache[comb] = self.ent([a, b])
        
        return self.ent_cache[a] + self.ent_cache[b] - self.ent_cache[comb]
        
    def count_cats(self, features, return_counts=False):
        if len(features) == 1:
            a = features[0]
            
            if a in self.label_count_cache: 
                return self.label_count_cache[a]
        else:
            comb = self.comb(features[0], features[1])
            if comb in self.label_count_cache:
                return self.label_count_cache[comb]

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

        return ratio_pdf    