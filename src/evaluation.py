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
    precision_rates = [0]*4
    recall_rates = [0]*4
    f1_measures = [0]*4

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
    unpruned_total_classification_rate = []
    pruned_total_classification_rate = []

    # Depths
    total_unpruned_depth = []
    total_pruned_depth = []

    # Loop for the k-fold cross validation
    for i in range(k):

        # We set the training set and the test set.
        training_set = np.vstack(np.delete(folds, np.s_[i], 0))
        test_set = folds[i]

        # Number of folds for the training set
        n = int(math.ceil(len(training_set) / len(test_set)))
        splitted_training_set = split_dataset(n, training_set)

        # We want to keep the best tree aftet the second n-fold cross validation
        best_pruned_tree = None
        best_score = math.inf

        # Loop for n-fold cross validation on training set for pruning tree.
        for j in range(n):

            # We set the sub-training set and the validation set.
            sub_training_set = np.vstack(np.delete(splitted_training_set, np.s_[j], 0))
            validation_set = splitted_training_set[j]

            # Then we call the function to create the pruned decision tree.
            tree, depth = decision_tree_training(sub_training_set)
            pruned_tree = prune(tree, sub_training_set, validation_set)
            matrix, classification_rate = evaluate(validation_set, pruned_tree)

            # If this model is better than the old one, we update it
            if (best_pruned_tree is None) or (best_score < classification_rate):
                best_score = classification_rate
                best_pruned_tree = pruned_tree

        unpruned_tree, unpruned_depth = decision_tree_training(training_set)

        # Now that we have the best tree, we can run the final evaluation
        unpruned_matrix, unpruned_classification_rate = evaluate(test_set, unpruned_tree)
        pruned_matrix, pruned_classification_rate = evaluate(test_set, best_pruned_tree)

        # We get the depth of each trees
        total_unpruned_depth.append(unpruned_depth)
        total_pruned_depth.append(best_pruned_tree.get_depth())

        # We increment our global confusion_matrices
        unpruned_confusion_matrix = np.add(unpruned_confusion_matrix, unpruned_matrix)
        pruned_confusion_matrix = np.add(pruned_confusion_matrix, pruned_matrix)

        # We increment the classification rates
        unpruned_total_classification_rate.append(unpruned_classification_rate)
        pruned_total_classification_rate.append(pruned_classification_rate)

    return unpruned_confusion_matrix, unpruned_total_classification_rate, total_unpruned_depth, \
           pruned_confusion_matrix, pruned_total_classification_rate, total_pruned_depth


# Show metrics
def show_metrics(metrics, title):

    # Unpruned Metrics
    unpruned_confusion_matrix = metrics[0]
    unpruned_total_classification_rate = metrics[1]
    unpruned_depth = metrics[2]

    # Pruned Metrics
    pruned_confusion_matrix = metrics[3]
    pruned_total_classification_rate = metrics[4]
    pruned_depth = metrics[5]

    # Unpruned tree
    unpruned_metrics = get_metrics(unpruned_confusion_matrix)
    print("######### " + title + " #########\n")
    print("Unpruned Metrics:\n")
    print("Confusion Matrix:")
    print(unpruned_confusion_matrix)
    print("\nMetrics:")

    output_matrix = [['      ', 'Precision', 'Recall   ', 'F1-measures']]
    # For each room
    for i in range(4):
        array = ["Room " + str(i + 1)]
        for value in unpruned_metrics.values():
            val = str(round(value[i], 4))
            while len(val) < 9:
                val += ' '
            array.append(val)
        output_matrix.append(array)
    output_matrix = np.array(output_matrix)

    # We print the matrix easy to read
    print(output_matrix)
    print("\nAverage Classification Rate: " +
          str(round((sum(unpruned_total_classification_rate) / len(unpruned_total_classification_rate)), 4)))
    print("-------------------------------")

    # Pruned tree
    pruned_metrics = get_metrics(pruned_confusion_matrix)
    print("Pruned Metrics:\n")
    print("Confusion Matrix:")
    print(pruned_confusion_matrix)
    print("\nMetrics:")

    output_matrix = [['      ', 'Precision', 'Recall   ', 'F1-measures']]
    # For each room
    for i in range(4):
        array = ["Room " + str(i + 1)]
        for value in pruned_metrics.values():
            val = str(round(value[i], 4))
            while len(val) < 9:
                val += ' '
            array.append(val)
        output_matrix.append(array)
    output_matrix = np.array(output_matrix)

    # We print the matrix easy to read
    print(output_matrix)
    print("\nAverage Classification Rate: " +
          str(round((sum(pruned_total_classification_rate) / len(pruned_total_classification_rate)), 4)))
    print("-------------------------------")

    # Depth comparisaon
    print("\nDepth Comparison:\n")
    print("Unpruned Tree:")
    print(unpruned_total_classification_rate)
    print(unpruned_depth)
    print("\nPruned Tree:")
    print(pruned_total_classification_rate)
    print(pruned_depth)


# Get results.
print("Running Clean Dataset...")
clean = fold_cross_validation(10, clean_dataset)
print("Running Noisy Dataset...")
noisy = fold_cross_validation(10, noisy_dataset)
show_metrics(clean, "Clean Dataset")
show_metrics(noisy, "Noisy Dataset")