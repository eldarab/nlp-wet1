from metodot_ezer import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def accuracy(true_file, predictions_file):
    # getting tags
    true_tags = get_file_tags(true_file)
    predicted_tags = get_file_tags(predictions_file)

    # calculating accuracy
    total_predictions = len(true_tags)
    errors = 0
    for true, pred in zip(true_tags, predicted_tags):
        if true != pred:
            errors += 1
    return errors / total_predictions


def confusion_matrix(true_file, predictions_file, errors_to_display=10, show=False, order='freq', slice_on_pred=True):
    """
    :param true_file: Some text file of the format "word_tag word_tag word_tag word_tag..."
    :param predictions_file: Some text file of the format "word_tag word_tag word_tag word_tag..."
    :param errors_to_display: Number of errors to display in CM, as required in the instructions
    :param show: Whether or not to show the CM at the end
    :param order: 'freq' for frequent errors left and up, 'lexi' for lexicographic order of tags
    :param slice_on_pred: The axis to slice errors on (can be either True to slice on true predictions axis or
                          False to slice on predicted labels axis)
    :return: DataFrame with the CM values, axes, rows and cols are labeled
    """
    # getting tags
    true_tags = get_file_tags(true_file)
    predicted_tags = get_file_tags(predictions_file)
    all_possible_tags = set(true_tags)

    # creating "raw" confusion matrix
    n = len(all_possible_tags)
    cm = pd.DataFrame(np.zeros((n, n)), columns=all_possible_tags, index=all_possible_tags)
    for true, pred in zip(true_tags, predicted_tags):
        cm.loc[true][pred] += 1

    # renaming axes names
    cm = cm.rename_axis('predicted label', axis='columns')
    cm = cm.rename_axis('true label', axis='rows')

    # if the user requested to slice on true labels axis, than we need to transpose the raw CM
    # and the logic applies with no change at all. It is set to check a negated if statement because original logic
    # was designed to make sense for slicing on prediction axis.
    if not slice_on_pred:
        cm = cm.transpose()

    # slicing rows/cols of confusion matrix to fit it to the exercise requirements
    # finding most common error
    total_tag_predictions = cm.sum(axis=1)
    correct_tag_predictions = cm.values.diagonal()
    tag_errors = total_tag_predictions - correct_tag_predictions
    tag_errors_sorted = tag_errors.sort_values(ascending=False)
    # slicing matrix
    top_errors = list(tag_errors_sorted.index.values)[:errors_to_display]
    cm = cm[top_errors]

    # reordering rows and cols by request
    if order == 'freq':
        cm = cm.reindex(top_errors, axis=1)
        cm = cm.reindex(tag_errors_sorted.index.values, axis=0)

    if order == 'lexi':
        cm = cm.reindex(sorted(list(cm.columns)), axis=1)
        cm = cm.reindex(sorted(list(cm.index.values)), axis=0)

    # plotting
    if show:
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        ax = sns.heatmap(cm, annot=True, cmap='Blues')
        plt.show()
    return cm


# TODO delete this after conflict with sklearn CM is solved
def raw_confusion_matrix(true_file, predictions_file):
    # getting tags
    true_tags = get_file_tags(true_file)
    predicted_tags = get_file_tags(predictions_file)
    all_possible_tags = set(true_tags)

    # creating "raw" confusion matrix
    n = len(all_possible_tags)
    cm = pd.DataFrame(np.zeros((n, n)), columns=all_possible_tags, index=all_possible_tags)
    for true, pred in zip(true_tags, predicted_tags):
        cm.loc[true][pred] += 1
    return cm.values
