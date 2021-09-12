# Big MRMR

Maximum Relevance Minimum Redundancy for big datasets.

This Python library uses Spark and Cython to speed up the calculations on big datasets.

This post in the Uber Engineering blog (https://eng.uber.com/optimal-feature-discovery-ml/) inspired me to develop this library.

In order to know more about MRMR read the previous post or go to the bottom of this README to find useful links.

You can find below the documentation and some code examples.

## Documentation

The main method you will need to use is the one from:

`from mrmr import MRMR`

More specifically, you will need to instantiate an object from the class `MRMR` and then invoke the function `mrmr()` (i.e. `MRMR(...).mrmr()`).


### MRMR class

You can configure the MRMR calculation via the instantiation parameters of the `MRMR` class. Here are the parameters:

`df`: `pyspark.sql.dataframe.DataFrame` : The input Apache Spark DataFrame with the data.

`target` : `string` : Name of the column that represents your target column. This column must be included in the input dataframe.

`k` (Default value `10`): `int` : Number of features extracted. More specifically, this sets the number of features with the highest MRMR. The default value is `10`, but if you have fewer variables, you must place here a lower value.

`subset` (Default value `[]`): `List[string]` : List of column names to use. These columns must be included in the input dataframe. The default value is an empty list and it will use all the columns.

`cont_vars` (Default value `[]`): `List[string]` : List of columns that are continuous variables. The columns in this list will have their values replaced by their decile, converting them in discrete variables. These columns must be included in the input dataframe. Consider this as an easy discretization, maybe you want to do your own discretization based on the meaning of your data.

`replace_na` (Default value `False`): `bool` : True if you want to have the NAs/Null/None values replaced or not. Numerical variables will have the value -1 when Null, and string variables will have the value "Null" when Null.

`optimal_k` (Default value `False`): `bool` : True if you want to calculate the best subset with size `k` of columns that, on average, they generate the highest MRMR. If False, it will calculate the MRMR of each column using all the columns in the dataframe (or the ones specified in the `subset` parameter).

What is the difference? If you look at MRMR formula, you will see that the Mutual Information of a variable is calculated taking into all the other variables. So, when `optimal_k` is True, it will only take into account the variables in a sample of size `k`. If False, it will take into account all variables.

This means, that when `optimal_k` is True the `mrmr()` method will return a DataFrame with samples of variables of size `k` and their average MRMR. When `optimal_k` is False a DataFrame with the name of the variables and the MRMR of each of them (only the top `k` variables will be returned).

`top_best_solutions` (Default value `5`): `int` : Number of top best variables subsets when `optimal_k` is True. When `optimal_k` is False, this parameter is ignored.

`must_included_vars` (Default value `[]`): `List[string]` : List of names of columns that must be included in the solutions. These columns must be included in the input dataframe and in the `subset` parameter, if used.

`max_mins` (Default value `None`): `float` : If different than `None`, the calculation will end after that number of minutes. The countdown will start at the beginning of the heaviest part of the code: the calculation of the optimal columns according to MRMR. It is advisable to set a value to this parameter, because depending on the number of columns of your DataFrame and on the value of `k`, it may take a very long time to explore all the possible solutions.

`cache_or_checkp` (Default value `None`): `string` : If different than `None`, it will cache the Spark DataFrame when its value equals `cache` and it will checkpoint the Spark DataFrame when its value equals `checkpoint`. Take into account that if you want to checkpoint the Spark DataFrame you will need to set a checkpoint directory (Google this or check the examples to see how).

`seed` (Default value `16121993`): `int` : Random seed value to get reproducible results. This is used only when `optimal_k` equals True, otherwise it is ignored.

#### mrmr function

`mrmr` : Returns a Pandas DataFrame. When `optimal_k` is True the `mrmr()` method will return a DataFrame with samples of variables of size `k` and their average MRMR. When `optimal_k` is False a DataFrame with the name of the variables and the MRMR of each of them (only the top `k` variables will be returned).

## Performance considerations

It is highly advisable to checkpoint or store the input DataFrame before using it in the `mrmr` function. This way, Spark will forget about the previous DAG and calculations will be faster. In the examples, you will see the input DataFrame is stored in Parquet format before using it. 

When `optimal_k` is False, the process is way less heavier than when equals to True. As a reference, I was able to run the algorithm when `optimal_k` is False in 36 minutes on a ~10GB with a 2015 laptop (2 cores, 4 with hyperthreading, 16GB RAM, HDD). 

When `optimal_k` is True, it depends on the size of the dataframe as well as on the number of combinations to try. If you are experiencing problems you will want to random sample your DataFrame in order to work with a smaller DataFrame. 

If you also struggle with performance, consider creating a Spark Session that runs on a cluster.

## Useful links

- What is MRMR: https://eng.uber.com/optimal-feature-discovery-ml/

- What is Mutual Information: https://colah.github.io/posts/2015-09-Visual-Information/#mutual-information

- Why Adjusted Mutual Information: this is used when `optimal_k` equals True in order to calculate comparable MRMR results in different variables subsets.

- How Adjusted Mutual Information is calculated:

https://en.wikipedia.org/wiki/Adjusted_mutual_information

Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance, JMLR

https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf

- What does corrected for chance mean: https://stats.stackexchange.com/questions/334045/whats-the-meaning-of-corrected-for-chance

