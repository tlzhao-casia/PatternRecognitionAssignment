# Feature Extraction

### Description
This is the codes for the fourth assignment of my class Pattern Recognition and Machine Learning.
The main purpose of this experiment is to study the 2 classical feature extraction
algorithms: PCA and LDA.

I implement PCA and LDA and apply them on the four popular datasets: iris, wine, vowel and OLHWDB1.1. 
OLHWDB1.1 is a rather large dataset for handwritten chinese character recognition with 898573 training 
samples and 224559 validation samples devided into 3755 classes. Because of the time constraint, I just 
select a tiny subset of OLHWDB1.1 with 200 classes to conduct my experiments.

### Feature visualization
To understand the differences and relationships between PCA, LDA and the original features intuitionally, 
I reduce the dimension of features for iris, wine and vowel dataset to 2 with PCA and LDA, respectively.
To do that, just simply run the following script:
```shell
python extract_first_two_dims.py
```
After that, the results will be written in `dataset_ori/pca/lda.feature`, for example, the PCA features 
of iris dataset will be written in `iris_ori.feature` The results are shown as follows:
![](jpgs/feature_visualization.jpg) 

### Feature extraction for classification
