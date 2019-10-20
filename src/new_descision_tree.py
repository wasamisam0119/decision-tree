import math
from scipy import stats
from scipy.special import factorial
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
from io import StringIO
import os

#
# with open('wifi_db/clean_dataset.txt', 'r') as f:
#     for line in f.readlines():
#         print(line)


class DecisionTree:
    # dataMatrix = np.loadtxt('wifi_db/clean_dataset.txt')

    def decision_tree_learning(self, dataMatrix, max_depth, pruneIndex):
        label = dataMatrix[:, dataMatrix.shape[1] - 1]
        H = entropy_calc(label)
        classes = np.unique(label)
        data_by_column = dataMatrix[:, 0:dataMatrix.shape[1] - 1]

        for idx in range(0, dataMatrix.shape[1] - 1):
            cur_col = dataMatrix[:, idx]
            sort_col = np.sort(cur_col)

    # label_counter is a dictionary that has
    # key: label
    # value: counter of the label in the dataset
    # {'yes':5}


    def information_gain(self, S_all, S_left, S_right):
        info_gain = self.entropy_calc(S_all) - remainder
        return info_gain

    def depth_calc(self):
        left_depth = self.left.depth_calc()
        right_depth = self.right.depth_calc()
        max_depth = max(left_depth, right_depth)
        return max_depth

class Test:
    @classmethod
    def test_entropy_calc(cls):
        test_dict = {1: 4, 2: 2, 3: 1, 4: 1}
        total_samples = 8
        H = entropy_calc(test_dict, total_samples)
        assert (H == 1.75)


def entropy_calc(label_counters, total_samples):
    entropy = 0
    # total_samples = sum(label_counters.values())
    base = 2

    # if n_labels < 1:
    #     return 0

    for label, samples in label_counters.items():
        probability = samples / total_samples
        entropy -= probability * math.log(probability, base)

    return entropy

#label counters is a dict that its key is the label value and its value is the count
def remainder_calc(len_S_left , left_label_counter,len_S_right, right_label_counters):
    total_len = len_S_left + len_S_right
    remainder = len_S_left / total_len * entropy_calc(left_label_counter, len_S_left) + len_S_right / total_len * entropy_calc(right_label_counters, len_S_right)
    return remainder


def gain(total_gain, len_S_left , left_label_counter,len_S_right, right_label_counter):
    new_gain = total_gain - remainder_calc(len_S_left , left_label_counter,len_S_right, right_label_counter)
    return new_gain


def read_dataset():
    filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/co553-cbc-dt/wifi_db/clean_dataset.txt')
    print(filename)
    dataMatrix = np.loadtxt(filename)
    print(dataMatrix)


def find_split(training_data):
    features = len(training_data[0]) - 1  #calculate number of features
    #calculate the unique value of label and its count
    unique, counts = np.unique(training_data[:,-1], return_counts=True)
    total_counts = dict(zip(unique, counts)) #form into a dict
    base_entropy = entropy_calc(total_counts, len(training_data))


    base_information_gain = 0.0;
    best_feature = -1
    best_feature_value = -1

    max_gain = 0
    for i in range(features):
        # the values of each feature eg. the height of a group of people [170,180,167]
        sorted_data = training_data[training_data[:, i].argsort()]

        feature_list = [item[i] for item in sorted_data]
        unique_values = set(feature_list)
        sorted_univalue = sorted(list(unique_values))
        new_entropy = 0.0
        # sorted_data = dataMatrix[dataMatrix[:, 1].argsort()]
        prev_val = -10000
        #initialize the left counters
        initial_l_count = []
        for label in unique:
            initial_l_count.add(0)
        left_counters = dict(zip(unique,initial_l_count))
        right_counters = total_counts


        for index,value in enumerate(feature_list):
            if prev_val == value:
                continue
            else:
                current_gain = gain(base_entropy,index,left_counters,len(feature_list),right_counters)

                #sorted_data[:,-1][:index]
                #sorted_data[:,-1][index:]
            prev_val = value
            if current_gain > max_gain:
                max_gain = current_gain
                best_feature = i
                best_feature_value = value

            left_counters[sorted_data[index][-1]]+=1
            right_counters[sorted_data[index][-1]]-=1

    return (best_feature,best_feature_value)










        infoGain = gain(base_entropy,)  # 计算当前分类的信息增益


        #if (infoGain > bestInfoGain):  # 比较那种分类的信息增益最大并返回
        #     bestInfoGain = infoGain
        #    bestFeature = i
    return bestFeature


def split_dataset(training_data,feature,value):


    pass

def desicion_tree_training(training_data,depth):

    if np.all(training_data[:,-1]==training_data[0][-1]):
        return Node(training_data[0][-1],depth)
    else:
        pass





filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/co553-cbc-dt/wifi_db/clean_dataset.txt')
dataMatrix = np.loadtxt(filename)
print(dataMatrix[:100,-1])
print(len(dataMatrix[0]))
print(dataMatrix[:100])
print(dataMatrix[dataMatrix[:, 1].argsort()][:100])