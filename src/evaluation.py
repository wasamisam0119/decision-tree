from decision_trees import *

clean_dataset = 'wifi_db/clean_dataset.txt'


def get_matrix_from_file(file):
    matrix = np.loadtxt(file)
    np.random.shuffle(matrix)
    return matrix


def split_dataset(nb_folds, file):
    dataset = get_matrix_from_file(file)
    len_fold = len(dataset) / nb_folds
    folds = []
    for i in range(nb_folds):
        start = int(i * len_fold)
        end = int(start + len_fold)
        folds.append(dataset[start:end])
    folds = np.array(folds)
    return folds


def fold_cross_validation(k, dataset):
    folds = split_dataset(k, dataset)
    confusion_matrix = np.array([[0]*4]*4)
    # Loop for the construction of the decision trees and evaluation
    for i in range(k):
        training = np.vstack(np.delete(folds, np.s_[i], 0))
        test = folds[i]

        # Call the function that build the tree
        tree, depth = decision_tree_training(training, 0)
        print(tree)

        # Iterate over the data of the fold test
        for row in test:
            features = row[:-1]
            label = row[-1]
            # Call the decision tree
            # Get the predicted room
            # Compare it with the label
            # Increment confusion_matrix



    incorrect = 0


fold_cross_validation(10, clean_dataset)