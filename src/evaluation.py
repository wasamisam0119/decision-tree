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


# Return the confusion matrix and the classification rate of the tree.
def evaluate(test_db, trained_tree):

    # Confusion matrix for this tree.
    confusion_matrix = np.array([[0]*4]*4)
    error = 0

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
    confusion_matrix = np.array([[0]*4]*4)

    # Total classification rate
    total_classification_rate = 0

    # Loop for the k-fold cross validation
    for i in range(k):

        # We set the training set and the test set.
        training_set = np.vstack(np.delete(folds, np.s_[i], 0))
        test_set = folds[i]

        # Number of folds for the training set
        n = int(math.ceil(len(training_set) / len(test_set)))
        training_set = split_dataset(n, training_set)

        # We want to keep the best tree aftet the second n-fold cross validation
        best_tree = None
        best_score = math.inf

        # Loop for n-fold cross validation on training set.
        for j in range(n):

            # We set the sub-training set and the validation set.
            sub_training_set = np.vstack(np.delete(training_set, np.s_[j], 0))
            validation_set = training_set[j]

            # Then we call the function to create the decision tree.
            tree, depth = decision_tree_training(sub_training_set)
            matrix, classification_rate = evaluate(validation_set, tree)

            # If this model is better than the old one, we update it
            if (best_tree is None) or (best_score < classification_rate):
                best_score = classification_rate
                best_tree = tree

        # Now that we have the best tree, we can run the final evaluation
        matrix, classification_rate = evaluate(test_set, best_tree)

        # We increment our global confusion_matrix
        confusion_matrix = np.add(confusion_matrix, matrix)
        total_classification_rate += classification_rate

    # Average classification rate
    average_classification_rate = total_classification_rate / k

    return confusion_matrix, average_classification_rate


clean_confusion_matrix, clean_accuracy = fold_cross_validation(10, clean_dataset)
print(clean_confusion_matrix)
print(get_metrics(clean_confusion_matrix))
print(clean_accuracy)
noisy_confusion_matrix, noisy_accuracy = fold_cross_validation(10, noisy_dataset)
print(noisy_confusion_matrix)
print(get_metrics(noisy_confusion_matrix))
print(noisy_accuracy)