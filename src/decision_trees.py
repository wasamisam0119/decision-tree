import math
import numpy as np
import os
from tree import Node


# class DecisionTree:
#     dataMatrix = np.loadtxt('wifi_db/clean_dataset.txt')
#
#     def decision_tree_learning(self, dataMatrix, max_depth, pruneIndex):
#         label = dataMatrix[:, dataMatrix.shape[1] - 1]
#         H = entropy_calc(label)
#         classes = np.unique(label)
#         data_by_column = dataMatrix[:, 0:dataMatrix.shape[1] - 1]
#
#         for idx in range(0, dataMatrix.shape[1] - 1):
#             cur_col = dataMatrix[:, idx]
#             sort_col = np.sort(cur_col)
#
#     label_counter is a dictionary that has
#     key: label
#     value: counter of the label in the dataset
#     {'yes':5}
#
#
#     def information_gain(self, S_all, S_left, S_right):
#         info_gain = self.entropy_calc(S_all) - remainder
#         return info_gain
#
#     def depth_calc(self):
#         left_depth = self.left.depth_calc()
#         right_depth = self.right.depth_calc()
#         max_depth = max(left_depth, right_depth)
#         return max_depth

class Test:
    @classmethod
    def test_entropy_calc(cls):
        test_dict = {1: 4, 2: 2, 3: 1, 4: 1}
        total_samples = 8
        H = entropy_calc(test_dict, total_samples)
        assert (H == 1.75)

    # tests that the tree is a correct decision tree from the training data
    @staticmethod
    def test_tree_on_training_data(tree, dataset):
        for datapoint in dataset:
            features = datapoint[:-1]
            prediction = predict(tree, features)
            assert datapoint[-1] == prediction


def entropy_calc(label_counters, total_samples):
    entropy = 0
    base = 2
    if total_samples < 1:
        return 0
    for label, samples in label_counters.items():
        probability = samples / total_samples
        if samples == 0:
            continue
        entropy -= probability * math.log(probability, base)
    return entropy


# label counters is a dict that its key is the label value and its value is the count
def remainder_calc(len_S_left, left_label_counter, len_S_right, right_label_counters):
    total_len = len_S_left + len_S_right
    remainder = len_S_left / total_len * entropy_calc(left_label_counter, len_S_left) + len_S_right / total_len * entropy_calc(right_label_counters, len_S_right)
    return remainder


def gain(total_gain, len_S_left , left_label_counter,len_S_right, right_label_counter):
    new_gain = total_gain - remainder_calc(len_S_left , left_label_counter,len_S_right, right_label_counter)
    return new_gain


def read_dataset():
    filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/co553-cbc-dt/wifi_db/clean_dataset.txt')
    print(filename)
    data_matrix = np.loadtxt(filename)
    print(data_matrix)


def find_split(training_data):
    features = len(training_data[0]) - 1  # calculate number of features
    # calculate the unique value of label and its count
    unique, counts = np.unique(training_data[:, -1], return_counts=True)
    # total_counts is a dictionary that keeps for each label how many times it appears in the dataset
    total_counts = dict(zip(unique, counts))  # form into a dict
    # base entropy is the total entropy of the dataset
    base_entropy = entropy_calc(total_counts, len(training_data))
    best_feature = -1
    best_value = -1
    best_value_position_in_sorted = -1
    max_gain = 0
    for i in range(features):
        # the values of each feature e.g. the height of a group of people [170,180,167]
        sorted_data = training_data[training_data[:, i].argsort()]
        feature_list = [datapoint[i] for datapoint in sorted_data]
        # initialize the left counters
        initial_l_count = []
        for _label in unique:
            initial_l_count.append(0)
        left_counters = dict(zip(unique, initial_l_count))
        right_counters = total_counts.copy()#why use copy?
        prev_val = -10000 # initial a value for feature_list[index-1]
        for index, value in enumerate(feature_list): #moving the pointer
            # if prev_val == value:
            #     continue

            if prev_val != value:
                current_gain = gain(base_entropy, index, left_counters, len(feature_list) - index, right_counters)
                if current_gain > max_gain:
                    max_gain = current_gain
                    best_feature = i
                    best_value = value
                    best_value_position_in_sorted = index
            prev_val = value
            left_counters[sorted_data[index][-1]] += 1
            right_counters[sorted_data[index][-1]] -= 1
    return best_feature, best_value, best_value_position_in_sorted


# split is the values returned by find_split
# the split_dataset function is to be called after finding a split, in order to actually split the dataset
def split_dataset(training_data, split):
    best_feature, _best_value, best_value_position_in_sorted = split
    sorted_data = training_data[training_data[:, best_feature].argsort()]
    left_dataset = sorted_data[:best_value_position_in_sorted, :]
    right_dataset = sorted_data[best_value_position_in_sorted:, :]
    # print(best_feature, _best_value, best_value_position_in_sorted)
    # print(left_dataset)
    # print(right_dataset)
    return left_dataset, right_dataset


# creates the decision tree
def decision_tree_training(training_data, depth=0):
    if np.all(training_data[:, -1] == training_data[0][-1]):
        return Node(depth=depth, label=training_data[0][-1]), depth
    else:
        # find where to split
        split = find_split(training_data)
        # split there
        #however, we can just record the position instead of actual split the dataset
        left_dataset, right_dataset = split_dataset(training_data, split)
        feature, value, _ = split
        # create node at this point
        node = Node(feature=feature, split_value=value, depth=depth)
        # create left and right branches of the decision tree
        left_node, left_depth = decision_tree_training(left_dataset, depth + 1)
        right_node, right_depth = decision_tree_training(right_dataset, depth + 1)
        # connect the node to the left and right branches
        node.set_left_node(left_node, left_depth)
        node.set_right_node(right_node, right_depth)
        return node, max(left_depth, right_depth)


def predict_datapoint(decision_tree: Node, datapoint):
    if decision_tree.is_leaf():
        # print('leaf', decision_tree.label)
        # a = decision_tree.label
        return decision_tree.label
    else:
        if datapoint[decision_tree.feature] < decision_tree.split_value:
            # print('left')
            return predict_datapoint(decision_tree.left_node, datapoint)
        else:
            # print('right')
            return predict_datapoint(decision_tree.right_node, datapoint)


def predict(decision_tree, X_test):
    Y_test = []
    for x in X_test:
        y = predict_datapoint(decision_tree,x)
        Y_test.append(y)
    return Y_test

'''
def train_test_split(training_data,percentage):
    number_of_data = training_data.shape[0]
    train_test_index = int(percentage * number_of_data)
    data_train, data_test = dataMatrix[0:train_test_index,:], dataMatrix[train_test_index:,:]
    return data_train,data_test


# filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/co553-cbc-dt/wifi_db/clean_dataset.txt')
filename = os.path.join(parentDirectory, 'dataset/co553-cbc-dt/wifi_db/clean_dataset.txt')
dataMatrix = np.loadtxt(filename)
print(dataMatrix)
np.random.shuffle(dataMatrix)

# here is call the recursive decision_tree_training to create the decision tree
tree, depth = decision_tree_training(dataMatrix, 0)
print("Depth:", depth)
print("Tree:", tree)
a = predict(tree, dataMatrix[0])
Test.test_tree_on_training_data(tree, dataMatrix)
'''