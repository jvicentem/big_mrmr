import os
import setuptools

from Cython.Build import cythonize
import numpy as np

with open('README.md', 'r') as fh:
    long_description = fh.read()

os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '30'

setuptools.setup(
    name='big_mrmr',
    version='1.0.0',
    author='José Vicente Mellado',
    author_email='contact@jvm16.xyz ',
    description='Maximum Relevance Minimum Redundancy for big datasets',
    keywords='mrmr feature selection bigdata',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jvicentem/big_mrmr',
    packages=setuptools.find_packages(),
    include_dirs=np.get_include(),    
    ext_modules=cythonize('./big_mrmr/cython_modules/_expected_mutual_info_fast.pyx'),    
    python_requires='>=3.6',
    install_requires=[
            'cython==0.29.24',
            'joblib==1.0.1',
            'numpy==1.19.5',
            'pandas==1.1.5',
            'pyspark>=3.1.2',
            'scipy==1.4.1',
            'scikit-learn==0.22.2.post1',
            'tqdm==4.62.0'
    ]
)