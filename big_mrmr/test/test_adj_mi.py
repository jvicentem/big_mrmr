import unittest

import numpy as np
import pandas as pd

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster._supervised import _generalized_average
from pyspark.sql import SparkSession


from big_mrmr.mrmr import MRMR
from big_mrmr.cython_modules._expected_mutual_info_fast import expected_mutual_information


class TestAdjustedMutualInfo(unittest.TestCase):
    def test_adj_mutual_info(self):       
        pdf = pd.DataFrame({'a': ['a', 'b', 'c', 'c'
                                 ], 
                            'b': ['a', 'b', 'c', 'b'
                                 ]})

        sk_result = adjusted_mutual_info_score(pdf['a'], pdf['b'])

        spark = (SparkSession.builder
                    .master('local[*]')
                    .appName('mrmr')
                    .getOrCreate()
                )

        mrmr_obj = MRMR(spark.createDataFrame(pdf), 
                        target='b', 
                        k=1)

        label_count = mrmr_obj.count_cats(['a', 'b'], return_counts=True)   

        classes = label_count['a'].unique()
        clusters = label_count['b'].unique()        

        aux_count_a = label_count[['a', 'count']]
        aux_count_b = label_count[['b', 'count']]

        emi = expected_mutual_information(aux_count_a.groupby('a').sum()['count'].values.astype('int32'), 
                                             aux_count_b.groupby('b').sum()['count'].values.astype('int32'), 
                                             mrmr_obj.df_count,
                                             len(classes), 
                                             len(clusters)
                                             )

        h_true = mrmr_obj.ent(['a'])
        h_pred = mrmr_obj.ent(['b'])
        
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
          
        result = (mrmr_obj.mi('a', 'b') - emi) / denominator      

        spark.stop()

        self.assertTrue(np.isclose(sk_result, result, atol=1e-10))

if __name__ == '__main__':
    unittest.main()