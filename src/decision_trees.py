import math
import numpy as np
import os
from tree import Node


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

    @staticmethod
    def test_pruning():
        testing_d = np.array([[45, 1], [50, 1], [37, 4], [68, 3], [90, 2]])
        tree, depth = decision_tree_training(testing_d, 0)
        assert depth == 2
        testing_v = np.array([[37, 1], [48, 1]])
        testing_v2 = np.array([[56, 1], [31, 1], [10, 4], [20, 4]])
        pruned = prune(tree, testing_d, testing_v)
        assert pruned.get_depth() == 0
        tree, depth = decision_tree_training(testing_d, 0)
        pruned2 = prune(tree, testing_d, testing_v2)
        assert pruned2.get_depth() == 2


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
    remainder = len_S_left / total_len * entropy_calc(left_label_counter,
                                                      len_S_left) + len_S_right / total_len * entropy_calc(
        right_label_counters, len_S_right)
    return remainder


def gain(total_gain, len_S_left, left_label_counter, len_S_right, right_label_counter):
    new_gain = total_gain - remainder_calc(len_S_left, left_label_counter, len_S_right, right_label_counter)
    return new_gain


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
        right_counters = total_counts.copy()
        prev_val = -10000  # initial a value for feature_list[index-1]
        for index, value in enumerate(feature_list):  # moving the pointer
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
def split_training_dataset(training_data, split):
    best_feature, _best_value, best_value_position_in_sorted = split
    sorted_data = training_data[training_data[:, best_feature].argsort()]
    left_dataset = sorted_data[:best_value_position_in_sorted, :]
    right_dataset = sorted_data[best_value_position_in_sorted:, :]
    return left_dataset, right_dataset


# creates the decision tree
def decision_tree_training(training_data, depth=0):
    if np.all(training_data[:, -1] == training_data[0][-1]):
        return Node(depth=depth, label=training_data[0][-1]), depth
    else:
        # find where to split
        split = find_split(training_data)
        # split there
        # however, we can just record the position instead of actual split the dataset
        left_dataset, right_dataset = split_training_dataset(training_data, split)
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


def prune(decision_tree: Node, training_dataset, validation_dataset):
    if decision_tree.is_leaf():
        return decision_tree
    # Split training and validation datasets into left and right, according to the feature value of the node
    left_training, right_training = pruning_split(training_dataset, decision_tree)
    left_v, right_v = pruning_split(validation_dataset, decision_tree)
    # Recursively prune left and right children-subtrees
    pruned_lnode = prune(decision_tree.left_node, left_training, left_v)
    pruned_rnode = prune(decision_tree.right_node, right_training, right_v)
    # Connect the new children nodes
    decision_tree.set_left_node(pruned_lnode, pruned_lnode.get_depth())
    decision_tree.set_right_node(pruned_rnode, pruned_rnode.get_depth())
    # Prune this node, if it is only connected to leaves
    if decision_tree.left_node.is_leaf() and decision_tree.right_node.is_leaf():
        # Assign label from the training data
        label = get_majority(training_dataset)
        new_tree = Node(label=label, depth=decision_tree.depth)
        _, full_score = evaluate(validation_dataset, decision_tree)
        _, pruned_score = evaluate(validation_dataset, new_tree)
        # Compare full and pruned trees
        if pruned_score >= full_score:
            return new_tree
        else:
            return decision_tree
    else:
        return decision_tree


# pruning_split is basically splitting the dataset based on value that stored in the current node
# which will be used in the prune() note: this is different
def pruning_split(dataset, tree: Node):
    feature = tree.feature
    split_value = tree.split_value
    left_list = []
    right_list = []
    for datapoint in dataset:
        if datapoint[feature] < split_value:
            left_list.append(datapoint)
        else:
            right_list.append(datapoint)
    return np.array(left_list), np.array(right_list)


# get the label that has the highest occurance
def get_majority(training_dataset):
    counts = np.bincount(training_dataset[:, -1].astype(int))  # cast to int to get the count
    return float(np.argmax(counts))  # return the value of the label


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
        y = predict_datapoint(decision_tree, x)
        Y_test.append(y)
    return Y_test


# Return the confusion matrix and the classification rate of the tree.
def evaluate(test_db, trained_tree):
    # Confusion matrix for this tree.
    confusion_matrix = np.array([[0] * 4] * 4)
    error = 0
    if len(test_db) == 0:
        return confusion_matrix, 1
    else:

        # Iterate over the data of the fold test
        for row in test_db:
            # We set the features and the label.
            features = row[:-1]
            label = int(row[-1])

            # Call the decision tree
            predicted_label = int(predict_datapoint(trained_tree, features))

            # Increment confusion_matrix
            confusion_matrix[label - 1][predicted_label - 1] += 1

            if predicted_label != label:
                error += 1

        classification_rate = 1 - (error / len(test_db))

        return confusion_matrix, classification_rate


