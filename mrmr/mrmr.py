from datetime import datetime
from modes.optimal_c import OptimalC

from pyspark.sql import functions as f
from pyspark.sql.window import Window

from modes.optimal_k import OptimalK


class MRMR:
    def __init__(self, df, target, k=10, subset=[], cont_vars=[], replace_na=False, 
                 optimal_k=False, top_best_solutions=5, must_included_vars=[], 
                 max_mins=10, cache_or_checkp=None):
        
        self.subset = df.columns if not subset else subset 

        if target not in self.subset:
            self.subset = self.subset + [target]

        self.df = df.select(*self.subset)
        self.replace_na = replace_na
        self.target = target
        self.k = k        
        self.optimal_k = optimal_k
        self.top_best_solutions = top_best_solutions
        self.must_included_vars = must_included_vars
                
        self.max_mins = max_mins if max_mins is not None else 999999                

        self._convert_to_categorical(cont_vars)             
                
        if cache_or_checkp is not None:
            if cache_or_checkp in 'cache':
                self.df = self.df.cache()
            elif cache_or_checkp in 'checkpoint':
                self.df = self.df.checkpoint()

        self.df_count = self.df.count()

    def mrmr(self):               
        print('MRMR calculation starting...')
        print(datetime.now())
        
        if self.optimal_k:
            opt = OptimalK(self.df,                     
                           self.replace_na, 
                           self.target, 
                           self.k, 
                           self.top_best_solutions, 
                           self.must_included_vars,
                           self.max_mins,
                           self.df_count)           
        else: 
            opt = OptimalC(self.df, 
                           self.replace_na, 
                           self.target, 
                           self.k, 
                           self.top_best_solutions, 
                           self.must_included_vars,
                           self.max_mins,
                           self.df_count)

        return opt.calculate_optimal_vars()

    def _convert_to_categorical(self, cont_vars, n_cats=10):
        for v in self.subset:
            if v in cont_vars:
                self.df = (self.df
                               .withColumn(v, 
                                           f.ntile(n_cats).over(Window.partitionBy(f.lit(1)).orderBy(v)))
                          )
                
                if self.replace_na:
                    self.df = self.df.na.fill(-1.0, subset=[v])                
            else:
                self.df = self.df.withColumn(v, f.col(v).cast('string'))
                
                if self.replace_na:
                    self.df = self.df.na.fill('Null', subset=[v])
