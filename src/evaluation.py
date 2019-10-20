import numpy as np

clean_dataset = open("clean_dataset.txt", 'r')

def get_matrix_from_file(file):
    matrix = []
    for line in file:
        matrix.append(line.split())
    matrix = np.array(matrix).astype(int)
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
        if i == 0:
            training = np.vstack((folds[1:]))
        elif i == (k-1):
            training = np.vstack((folds[:-1]))
        else:
            training = np.vstack((folds[0:i], folds[i+1:]))

        test = folds[i]

        # Call the function that build the tree

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