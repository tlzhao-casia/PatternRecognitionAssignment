# Parametric & Non-Parametric Classification

### Dependency

* pytorch
* numpy

### Description

This is the codes of the first assignment of my class: Pattern Recognition and Machine Learning. 
I implement five classifiers (QDF, LDF, MQDF, RDA and Parzen window), and evaluate 
these five classifiers on 3 datasets (iris, wine and wdbc). Considering that for 
wine and wdbc, the dimension of features is high, while data samples are limited, which
makes it difficult to approximate the probability density function with these samples, 
so for these 2 datasets, I first normalize the data with their means and variances,
and then compress the data to 4 dimensions with PCA.

### Usage

You can run the train.py script to evaluate these different algorithms on sevral 
datasets. An example is shown as follows:
```shell
python train.py --cls qdf -d iris
``` 
This command will evaluate the QDF algorithm on iris dataset. More avaliable parameters 
and details are listed as follows:
* **--cls**: The classification algorithm, can be one of qdf, ldf, mqdf, rda, parzen
* **-d**: The evaluated dataset, can be one of iris, wine, wdbc
* **--val-rate**: The percentage of test samples
* **-k**: The number of singular values to be reserved for MQDF
* **--h**: The width of parzen window
* **--gamma**: The gamma parameter for RDA
* **--beta**: The beta parameter for RDA 
