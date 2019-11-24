### Semi Supervised Learning

### Description
This is the third assiginment of my class Parattern Recognition and Machine Learning. 
I implement 2 graph based semi-supervised learning algorithms (GRF and LLGC) and 
evaluate these two algorithms on 3 datasets (MNIST, vowel and sonar). For MNIST dataset,
I randomly select 2000 samples from the whole training set, which contains 60000
images, to conduct my experiments. For each algorithm, I randomly select 20%, 30%
and 40% of all the samples as labeled data, predict the labels of the remaining 
samples and compare the predicted labels to their true counterparts. The accuaracy 
of prediction is reported for evaluation of different algorithms.

### Usage
The script is very easy to use. For example, if you want to evaluate GRF algorithm 
on MNIST dataset, you can simply enter this directory and run the following command:
```shell
python main.py -d MNIST -g grf -s 1 -r 0.3
```

Parameters and descriptions are listed as follows:
* **--dataset, -d:** STRING | The evaluated dataset, can be one of {MNIST, sonar, vowel}, default is MNIST.
* **--graph, -g:** STRING | The evaluated graph-based algorithm, can be one of {grf, llgc}, default is grf.
* **--rate, -r:** FLOAT | The percentage of labeled data, must be greater than 0 and less than 1, default is 0.3.
* **--sigma, -s** FLOAT | The sigma parameter for computation of weight matrix, default is 1.
* **--alpha, -a** FLOAT | The alpha parameter for LLGC, default is 0.1.
