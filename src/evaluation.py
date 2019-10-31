from src.decision_trees import *

clean_dataset = 'wifi_db/clean_dataset.txt'
noisy_dataset = 'wifi_db/noisy_dataset.txt'


# Shuffle the data and return it as a matrix.
def get_matrix_from_file(file):
    matrix = np.loadtxt(file)
    np.random.shuffle(matrix)
    return matrix


# Split the dataset into different folds.
def split_dataset(nb_folds, dataset):
    len_fold = len(dataset) / nb_folds
    folds = []
    for i in range(nb_folds):
        start = int(i * len_fold)
        end = int(start + len_fold)
        folds.append(dataset[start:end])
    folds = np.array(folds)
    return folds


# Return the metrics linked to the confusion matrix
def get_metrics(confusion_matrix):
    precision_rates = recall_rates = f1_measures = [0]*4

    for i in range(4):
        precision_rates[i] = confusion_matrix[i][i] / confusion_matrix.sum(axis=0)[i]
        recall_rates[i] = confusion_matrix[i][i] / confusion_matrix.sum(axis=1)[i]
        f1_measures[i] = (2 * precision_rates[i] * recall_rates[i]) / (precision_rates[i] + recall_rates[i])

    metrics = {'precision_rates': precision_rates, 'recall_rates': recall_rates, 'f1_measures': f1_measures}

    return metrics


# Run the k-fold cross validation and return the confusion matrix.
def fold_cross_validation(k, file):

    # We get the dataset from the file
    dataset = get_matrix_from_file(file)

    # We split the dataset in k-folds
    folds = split_dataset(k, dataset)

    # This is the final confusion matrix.
    unpruned_confusion_matrix = np.array([[0]*4]*4)
    pruned_confusion_matrix = np.array([[0]*4]*4)

    # Total classification rate
    unpruned_total_classification_rate = 0
    pruned_total_classification_rate = 0

    # Loop for the k-fold cross validation
    for i in range(k):

        # We set the training set and the test set.
        training_set = np.vstack(np.delete(folds, np.s_[i], 0))
        test_set = folds[i]

        # Number of folds for the training set
        n = int(math.ceil(len(training_set) / len(test_set)))
        training_set = split_dataset(n, training_set)

        # We want to keep the best tree aftet the second n-fold cross validation
        best_pruned_tree = None
        best_score = math.inf
        best_depth = 0

        # Loop for n-fold cross validation on training set for pruning tree.
        for j in range(n):

            # We set the sub-training set and the validation set.
            sub_training_set = np.vstack(np.delete(training_set, np.s_[j], 0))
            validation_set = training_set[j]

            # Then we call the function to create the pruned decision tree.
            tree, depth = decision_tree_training(sub_training_set)
            pruned_tree = prune(tree, sub_training_set, validation_set)
            pruned_depth = pruned_tree.get_depth()
            matrix, classification_rate = evaluate(validation_set, pruned_tree)

            # If this model is better than the old one, we update it
            if (best_pruned_tree is None) or (best_score < classification_rate):
                best_score = classification_rate
                best_pruned_tree = pruned_tree
                best_depth = pruned_depth

        unpruned_tree, unpruned_depth = decision_tree_training(training_set)

        # Now that we have the best tree, we can run the final evaluation
        unpruned_matrix, unpruned_classification_rate = evaluate(test_set, unpruned_tree)
        pruned_matrix, pruned_classification_rate = evaluate(test_set, best_pruned_tree)

        # We increment our global confusion_matrix
        unpruned_confusion_matrix = np.add(unpruned_confusion_matrix, unpruned_matrix)
        pruned_confusion_matrix = np.add(pruned_confusion_matrix, pruned_matrix)

        # We increment the classification rate
        unpruned_total_classification_rate += unpruned_classification_rate
        pruned_total_classification_rate += pruned_classification_rate

    # Average classification rate
    unpruned_average_classification_rate = unpruned_total_classification_rate / k
    pruned_average_classification_rate = pruned_total_classification_rate / k

    return unpruned_confusion_matrix, unpruned_average_classification_rate, \
           pruned_confusion_matrix, pruned_average_classification_rate


# Get results.
clean_confusion_matrix, clean_accuracy, clean_confusion_matrix_pruned, clean_accuracy_pruned = fold_cross_validation(10, clean_dataset)
print(clean_confusion_matrix)
print(get_metrics(clean_confusion_matrix))
print(clean_accuracy)
noisy_confusion_matrix, noisy_accuracy, noisy_confusion_matrix_pruned, noisy_accuracy_pruned = fold_cross_validation(10, noisy_dataset)
print(noisy_confusion_matrix)
print(get_metrics(noisy_confusion_matrix))
print(noisy_accuracy)