from datetime import datetime
import os
import sys
import time

import pandas as pd

sys.path.append('/home/jose/big_mrmr/mrmr')

os.chdir('./mrmr/cython_modules')
from Cython.Build import cythonize
os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '30'
cythonize('_expected_mutual_info_fast.pyx')
os.chdir('../..')

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
            .config('spark.sql.parquet.compression.codec', 'snappy')
            # .config("spark.scheduler.allocation.file", "/home/jose/mrmr/scheduling.xml")
            .getOrCreate()
       )

sc = spark.sparkContext

sc.setCheckpointDir('/home/jose/Desktop/checkpoints')

sc.setLocalProperty("spark.scheduler.pool", "mrmr")

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType

# df = spark.read.csv('/home/jose/big_mrmr/dataset/used_cars_data.csv', 
#                     header = True, 
#                     mode = 'DROPMALFORMED'
#                     )

# df = (df.drop(*['bed', 'bed_height', 'bed_length', 'cabin', 'combine_fuel_economy',
#                 'dealer_zip', 'city', 'description', 'is_certified', 'is_cpo', 'is_oemcpo',
#                 'latitude', 'listed_date', 'listing_id', 'longitude', 'main_picture_url',
#                 'major_options', 'model_name', 'power', 'sp_id', 'sp_name', 'torque',
#                 'transmission', 'transmission_display', 'trimId', 'trim_name',
#                 'vehicle_damage_category'
#                 ])
#         .withColumn('fleet', f.col('fleet').cast('string'))    
#         .withColumn('frame_damaged', f.col('frame_damaged').cast('string'))  
#         .withColumn('franchise_dealer', f.col('franchise_dealer').cast('string'))          
#         .withColumn('has_accidents', f.col('has_accidents').cast('string'))    
#         .withColumn('isCab', f.col('isCab').cast('string'))    
#         .withColumn('is_new', f.col('is_new').cast('string'))    
#         .withColumn('salvage', f.col('salvage').cast('string'))    
#         .withColumn('theft_title', f.col('theft_title').cast('string'))    
#         .withColumn('back_legroom', f.when(f.col('back_legroom').contains('in'),
#                                            f.regexp_replace('back_legroom', 'in', '')
#                                            ).cast('float')
#                     )
#         .withColumn('front_legroom', f.when(f.col('front_legroom').contains('in'),
#                                            f.regexp_replace('front_legroom', 'in', '')
#                                            ).cast('float')
#                     )
#         .withColumn('fuel_tank_volume', f.when(f.col('fuel_tank_volume').contains('gal'),
#                                            f.regexp_replace('fuel_tank_volume', 'gal', '')
#                                            ).cast('float')
#                     )
#         .withColumn('height', f.when(f.col('height').contains('in'),
#                                            f.regexp_replace('height', 'in', '')
#                                            ).cast('float')
#                     )
#         .withColumn('length', f.when(f.col('length').contains('in'),
#                                            f.regexp_replace('length', 'in', '')
#                                            ).cast('float')
#                     )
#         .withColumn('wheelbase', f.when(f.col('wheelbase').contains('in'),
#                                            f.regexp_replace('wheelbase', 'in', '')
#                                            ).cast('float')
#                     )
#         .withColumn('width', f.when(f.col('width').contains('in'),
#                                            f.regexp_replace('width', 'in', '')
#                                            ).cast('float')
#                     )
#         .withColumn('city_fuel_economy', f.col('city_fuel_economy').cast('float'))
#         .withColumn('daysonmarket', f.col('daysonmarket').cast('float'))
#         .withColumn('engine_displacement', f.col('engine_displacement').cast('float'))
#         .withColumn('highway_fuel_economy', f.col('highway_fuel_economy').cast('float'))
#         .withColumn('horsepower', f.col('horsepower').cast('float'))
#         .withColumn('mileage', f.col('mileage').cast('float'))
#         .withColumn('owner_count', f.col('owner_count').cast('float'))
#         .withColumn('savings_amount', f.col('savings_amount').cast('float'))
#         .withColumn('maximum_seating', f.when(f.col('maximum_seating').contains('seats'),
#                                               f.regexp_replace('maximum_seating', 'seats', '')
#                                              ).cast('float')
#                    )        
#         .withColumn('price', f.col('price').cast('float'))                                                                                                                                                                 
#         )

# df.write.parquet('/home/jose/big_mrmr/dataset/used_cars_data_custom', mode='overwrite')

df = (spark
       .read
       .parquet('/home/jose/big_mrmr/dataset/used_cars_data_custom')           
      )

mrmr_obj = MRMR(df.limit(1000000), 
                target='price', 
                k=5, 
                subset=['city_fuel_economy', 'daysonmarket', 'engine_displacement',
                        'highway_fuel_economy', 'horsepower', 'mileage', 'owner_count',
                        'savings_amount', 'seller_rating', 'back_legroom',
                        'front_legroom', 'fuel_tank_volume', 'height',
                        'length', 'wheelbase', 'width', 
                        'body_type', 'engine_cylinders', 'fleet', 'frame_damaged',
                        'franchise_dealer', 'fuel_type', 'has_accidents',
                        'isCab', 'is_new', 'maximum_seating', 'salvage',
                        'wheel_system', 'wheel_system_display', 
                        'price'
                        ],
                cont_vars=['city_fuel_economy', 'daysonmarket', 'engine_displacement',
                           'highway_fuel_economy', 'horsepower', 'mileage', 'owner_count',
                           'price', 'savings_amount', 'seller_rating', 'back_legroom',
                           'front_legroom', 'fuel_tank_volume', 'height',
                           'length', 'wheelbase', 'width'
                           ], 
                replace_na=True,
                optimal_k=True, 
                top_best_solutions=4, 
                must_included_vars=[],
                max_mins=120,
                cache_or_checkp=None)
    
start_time = time.time()
mrmr = mrmr_obj.mrmr()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

print('~~~~~~~~~~~~~~~~ END ~~~~~~~~~~~~~~~')
print('--- %s minutes ---' % ((time.time() - start_time) / 60))
print(datetime.now())
# print(mrmr.to_string())
print(mrmr)