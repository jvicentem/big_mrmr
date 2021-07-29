from datetime import datetime
import os
import sys
import time

import pandas as pd

sys.path.append('/home/jose/big_mrmr/mrmr')

os.chdir('./mrmr')
from Cython.Build import cythonize
os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '30'
cythonize('_expected_mutual_info_fast.pyx')
os.chdir('..')

import pyximport; pyximport.install()

from mrmr import MRMR

os.environ['SPARK_HOME'] = '/home/jose/anaconda3/envs/python36/lib/python3.6/site-packages/pyspark/'

import findspark
findspark.init() 

from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = (SparkSession.builder
            .master('local[*]')
            .appName('mrmr')
            .config(key = 'spark.driver.cores', value = '4')
            .config(key = 'spark.driver.memory', value = '10G')
            .config(key = 'spark.executor.memory', value = '10G') 
            .config(key = 'spark.driver.maxResultSize', value = '2G')
            .config('spark.scheduler.mode', 'FAIR')
            .config('spark.sql.execution.arrow.pyspark.enabled', 'true')  
            # .config("spark.scheduler.allocation.file", "/home/jose/mrmr/scheduling.xml")
            .getOrCreate()
       )

sc = spark.sparkContext

sc.setLocalProperty("spark.scheduler.pool", "mrmr")

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType

df = spark.read.csv('/home/jose/spark_review_2020/Parking_Violations_Issued_-_Fiscal_Year_2017.csv', 
               header = True, 
               mode = 'DROPMALFORMED', 
               schema = StructType([StructField('Summons Number', IntegerType(), True), 
                                    StructField('Plate ID', StringType(), True), 
                                    StructField('Registration State', StringType(), True),
                                    StructField('Plate Type', StringType(), True),
                                    StructField('Issue Date', TimestampType(), True),                                    
                                    StructField('Violation Code', IntegerType(), True),                                    
                                    StructField('Vehicle Body Type', StringType(), True),
                                    StructField('Vehicle Make', StringType(), True),                                    
                                    StructField('Issuing Agency', StringType(), True),                                    
                                    StructField('Street Code1', IntegerType(), True),                                    
                                    StructField('Street Code2', IntegerType(), True),
                                    StructField('Street Code3', IntegerType(), True),                                    
                                    StructField('Vehicle Expiration Date', IntegerType(), True),                                    
                                    StructField('Violation Location', StringType(), True),                                    
                                    StructField('Violation Precinct', IntegerType(), True),                                    
                                    StructField('Issuer Precinct', IntegerType(), True),                                    
                                    StructField('Issuer Code', IntegerType(), True),                                    
                                    StructField('Issuer Command', StringType(), True),                                    
                                    StructField('Issuer Squad', StringType(), True),                                    
                                    StructField('Violation Time', StringType(), True),                                    
                                    StructField('Time First Observed', StringType(), True),                                    
                                    StructField('Violation County', StringType(), True),                                    
                                    StructField('Violation In Front Of Or Opposite', StringType(), True),                                    
                                    StructField('House Number', StringType(), True),                                    
                                    StructField('Street Name', StringType(), True),                                    
                                    StructField('Intersecting Street', StringType(), True),                                    
                                    StructField('Date First Observed', IntegerType(), True),                                    
                                    StructField('Law Section', IntegerType(), True),                                    
                                    StructField('Sub Division', StringType(), True),                                    
                                    StructField('Violation Legal Code', StringType(), True),                                    
                                    StructField('Days Parking In Effect', StringType(), True),
                                    StructField('From Hours In Effect', StringType(), True),                                    
                                    StructField('To Hours In Effect', StringType(), True),                                    
                                    StructField('Vehicle Color', StringType(), True),                                    
                                    StructField('Unregistered Vehicle?', StringType(), True),                                    
                                    StructField('Vehicle Year', IntegerType(), True),                                    
                                    StructField('Meter Number', StringType(), True),                                    
                                    StructField('Feet From Curb', IntegerType(), True),                                    
                                    StructField('Violation Post Code', StringType(), True),                                    
                                    StructField('Violation Description', StringType(), True),                                    
                                    StructField('No Standing or Stopping Violation', StringType(), True),                                    
                                    StructField('Hydrant Violation', StringType(), True),                                    
                                    StructField('Double Parking Violation', StringType(), True)
                                   ]),
               timestampFormat='MM/dd/yyyy'
              )

df = (df.withColumn('Violation Code', f.col('Violation Code').cast('string'))
        .drop(*['Street Code1', 'Street Code2', 'Street Code3', 'Issuer Code', 
                'Violation Precinct', 'Issue Date']))
    
# mrmr_obj = MRMR(df, 
#                 target='Violation Code', 
#                 k=20, 
#                 subset=[], 
#                 cont_vars=['Summons Number',
#                            'Vehicle Expiration Date', 
#                            'Date First Observed', 'Feet From Curb'], 
#                 replace_na=True,
#                 optimal_k=False,
#                 max_mins=120)

mrmr_obj = MRMR(df, 
                target='Violation Code', 
                k=5, 
                subset=[],
                # subset=['No Standing or Stopping Violation',
                #         'Hydrant Violation',
                #         'Double Parking Violation'
                # ], 
                # subset=[#'Plate Type',
                #         'Law Section',
                #         'Violation County',
                #         'Issuer Squad',
                #         #'Sub Division',
                #         'Feet From Curb',
                #         #'Violation Description'
                #         ],
                # subset=['Summons Number', 'Vehicle Expiration Date', 
                #         'Date First Observed', 'Feet From Curb', 'Law Section',
                #         'Violation County', 'Issuer Squad', 'Hydrant Violation',
                #         'Double Parking Violation', 'No Standing or Stopping Violation',
                #         'Vehicle Color', 'Vehicle Year'
                #         ],
                cont_vars=['Summons Number', 'Vehicle Expiration Date', 
                           'Date First Observed', 'Feet From Curb'], 
                replace_na=True,
                optimal_k=True, 
                top_best_solutions=4, 
                must_included_vars=[],
                max_mins=120)
    
start_time = time.time()
mrmr, debug_mrmr = mrmr_obj.mrmr()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

print('~~~~~~~~~~~~~~~~ END ~~~~~~~~~~~~~~~')
print('--- %s minutes ---' % ((time.time() - start_time) / 60))
print(datetime.now())
# print(mrmr.to_string())
print(mrmr)