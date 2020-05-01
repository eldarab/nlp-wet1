from log_linear_memm import Log_Linear_MEMM
from time import strftime, time
from evaluation import *
from emailer import send_email
from sklearn import metrics  # TODO do not lehagish


def notify_email(start_time, preprocess_time, optimization_time, prediction_time):
    message_body = 'Start: ' + start_time + '\nPreprocess end: ' + preprocess_time + '\nOptimization end: ' + \
                   optimization_time + '\nPrediction end: ' + prediction_time
    send_email('eldar.abraham@gmail.com', 'Pa$$w0rd2019', ['eldar.a@campus.technion.ac.il'], message_body)


# TODO do not lehagish
def clean_tags(input_data):
    with open(input_data, 'r') as in_file:
        with open(input_data[:-5] + '_clean.txt', 'w') as out_file:
            for line in in_file:
                words_tags = line.split()
                for word_tag in words_tags:
                    word = word_tag.split('_')[0]
                    out_file.write(word + ' ')
                out_file.write('\n')


if __name__ == '__main__':
    start_time = strftime("%Y-%m-%d_%H-%M-%S")
    train_data = 'data/train1.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)

    #   Preprocessing
    # TODO maybe add all hyper-parameters as instance variables of the model? self.threshold for example
    threshold = 10
    model.preprocess(threshold=threshold)
    preprocess_time = strftime("%Y-%m-%d_%H-%M-%S")

    #   Optimizing / loading pre-trained weights
    optimization_time = strftime("%Y-%m-%d_%H-%M-%S")
    lam = 0
    maxiter = 50
    model.optimize(lam=lam,
                   maxiter=maxiter,
                   weights_path='dumps/weights_' + train_data[5:-5] + '_threshold=' + str(threshold) + '_lam=' +
                                str(lam) + '_iter=' + str(maxiter) + '_' + start_time + '.pkl')
    # model.load_weights()

    #   Predict
    # TODO evaluate with different beam sizes
    # prediction = model.predict('data/debugging_dataset_201_210_clean.txt')
    prediction_time = strftime("%Y-%m-%d_%H-%M-%S")

    #   Evaluation
    # true_file = 'data/debugging_dataset_201_210.wtag'
    # predictions_file = 'data/debugging_dataset_201_210_clean_predictions.txt'
    # true_tags = get_file_tags(true_file)
    # predicted_tags = get_file_tags(predictions_file)
    # confusion_matrix(true_file, predictions_file, show=False, slice_on_pred=False, order='freq')
    # accuracy = accuracy(true_file, predictions_file)

    notify_email(start_time, preprocess_time, optimization_time, prediction_time)

    # old testers
    # true_tags_set, predicted_tags_set = set(), set()
    # for tag1, tag2 in zip(true_tags, predicted_tags):
    #     if tag1 not in true_tags_set:
    #         true_tags_set.add(tag1)
    #     if tag2 not in predicted_tags_set:
    #         predicted_tags_set.add(tag2)
    # universal_set = true_tags_set.union(predicted_tags_set)
    # cm_eldar = raw_confusion_matrix(true_file, predictions_file)
    # cm_sklearn = metrics.confusion_matrix(true_tags, predicted_tags)
    # cm_eldar_sum_row = sorted([int(x) for x in cm_eldar.sum(axis=0)], reverse=True)
    # cm_eldar_sum_col = sorted([int(x) for x in cm_eldar.sum(axis=1)], reverse=True)
    # cm_eldar_sum_all = cm_eldar.sum()
    # cm_sklearn_sum_row = sorted(list(np.array(cm_sklearn).sum(axis=0)), reverse=True)
    # cm_sklearn_sum_col = sorted(list(np.array(cm_sklearn).sum(axis=1)), reverse=True)
    # cm_sklearn_sum_all = cm_sklearn.sum()