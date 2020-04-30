from sklearn.metrics import accuracy_score, confusion_matrix
from metodot_ezer import *


def accuracy(true_file, predicted_file):
    true_labels = get_all_labels(true_file)
    predicted_labels = get_all_labels(predicted_file)
    return accuracy_score(true_labels, predicted_labels)


def cm(true_file, predicted_file):
    true_labels = get_all_labels(true_file)
    predicted_labels = get_all_labels(predicted_file)
    return confusion_matrix(true_labels, predicted_labels)