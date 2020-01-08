## Predicting Indoor Position based on Wifi Signal Using Decision Tree ID3 Algorithm

In this project, we were asked to investigate and implement a decision tree algorithm that will be used to determine the indoor locations based on the seven WIFI signals strengths collected from a mobile phone.

The picture below is a more intuitive demonstration:

<img src="https://github.com/wasamisam0119/decision-tree/blob/master/src/Snip20200108_1.png" alt="ceremony" align=center>

The data can be found in the `dataset` folder

To tackle this problem, we implemented a decision tree algorithm that also comes with a 10-fold cross validation and bottom-up pruning. The details of this algorithm and evaluation will be given below. Before pruning, we have accuracies of **97.6%** and **80.05%** on the clean and noisy dataset, before puning, and **96.8%** and **88.2%** after pruning, respectively.

## Run

`python3 evaluation.py 'path_to_the_directory_of_the_dataset'`

It will then output ('Running...').
At the end, it will ouput all the metrics necessary to evaluate the performance of the tree without pruning and with pruning.