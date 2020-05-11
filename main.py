from log_linear_memm import Log_Linear_MEMM
import pickle
from auxiliary_functions import *


# f102_dict = model.feature_statistics.f102_count_dict
# f102_dict_sorted = {k: v for k, v in sorted(f102_dict.items(), key=lambda item: item[1], reverse=True)}


def cv1():
    train_path = 'data/train1.wtag'  # train1.wtag
    model_index = 100  # starting from big number to make sure we dont confuse with previously trained models
    with open('dumps/report_02-05-2020.txt', 'w') as report:
        report.write('---CV Report 02-05-2020---\n\n')
    thresholds = [5, 10, 15, 20]
    fix_thresholds = [100, 200, 250]
    lambdas = [0.1, 0.5, 0.7]
    for threshold in thresholds:
        for fix_threshold in fix_thresholds:
            for lam in lambdas:
                accuracy_test, accuracy_comp = evaluate(train_path, model_index, threshold, fix_threshold, lam)
                with open('dumps/report_02-05-2020.txt', 'a') as report:
                    report.write('model_' + str(model_index) + ':\n')
                    report.write('-threshold=' + str(threshold) + '\n')
                    report.write('-fix_threshold=' + str(fix_threshold) + '\n')
                    report.write('-lambda=' + str(lam) + '\n')
                    report.write('test1 accuracy=' + str(accuracy_test) + '\n')
                    report.write('comp1 accuracy=' + str(accuracy_comp) + '\n\n')
                model_index += 1
                print('Finished evaluating model index: ' + str(model_index))


def cv2(train_path, report_path):
    model_index = 202  # starting from big number to make sure we dont confuse with previously trained models
    thresholds = [7, 5, 3]
    fix_thresholds = [100, 70, 50]
    lambdas = [1, 0.5, 0.1, 0.01]
    maxiter = 100
    fix_weights = (1, 1, 1, 1)
    for threshold in thresholds:
        for fix_threshold in fix_thresholds:
            for lam in lambdas:
                accuracy_test, accuracy_comp, iterations = evaluate(train_path, model_index, threshold, fix_threshold,
                                                                    lam, maxiter, fix_weights)
                write_report_line(report_path, model_index, threshold, fix_threshold, lam, maxiter, iterations,
                                  fix_weights, acc_comp=accuracy_comp, acc_test=accuracy_test)
                model_index += 1
                print('model' + str(model_index) + 'finished training with test accuracy=' + str(accuracy_test))


def cv_small1(train_path, report_path):
    model_index = 300



def train_best(train_path, report_path):
    model_index = 200
    accuracy_test, accuracy_comp, iterations = evaluate(train_path, model_index, threshold=5, fix_threshold=100,
                                                        lam=0.5, maxiter=200, fix_weights=(1, 1, 1, 1))
    write_report_line(report_path, model_index, threshold=5, fix_threshold=100, lam=0.5, maxiter=200,
                      iterations=iterations, fix_weights=(1, 1, 1, 1), acc_comp=accuracy_comp, acc_test=accuracy_test)
    print('model' + str(model_index) + 'finished training with test accuracy=' + str(accuracy_test))
    model_index += 1
    accuracy_test, accuracy_comp, iterations = evaluate(train_path, model_index, threshold=5, fix_threshold=100,
                                                        lam=0.5, maxiter=100, fix_weights=(1, 1, 1, 1))
    write_report_line(report_path, model_index, threshold=5, fix_threshold=100, lam=0.5, maxiter=100,
                      iterations=iterations, fix_weights=(1, 1, 1, 1), acc_comp=accuracy_comp, acc_test=accuracy_test)
    print('model' + str(model_index) + 'finished training with test accuracy=' + str(accuracy_test))


def evaluate(train_path, model_index, threshold, fix_threshold, lam, maxiter, fix_weights):
    # creating and training a model
    model = Log_Linear_MEMM(threshold, fix_threshold, lam, maxiter, fix_weights)
    model.fit(train_path)  # using default maxiter=200
    iterations = model.iter

    # saving trained model
    model.save('model' + str(model_index))

    # predicting and saving predictions
    predictions_test1 = model.predict('data/test1.wtag', beam_size=2)  # test1.wtag
    with open('dumps/model' + str(model_index) + '_test1_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_test1, f)
    predictions_comp1 = model.predict('data/comp1_nltk_tagged.wtag', beam_size=2)  # comp1_nltk_tagged.wtag
    with open('dumps/model' + str(model_index) + '_comp1_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_comp1, f)

    # returning accuracy on test1.wtag and comp1_nltk_tagged.wtag
    accuracy_test1 = Log_Linear_MEMM.accuracy('data/test1.wtag', predictions_test1)  # test1.wtag
    accuracy_comp1_tagged = Log_Linear_MEMM.accuracy('data/comp1_nltk_tagged.wtag', predictions_comp1)  # comp1_XX.wtag
    return accuracy_test1, accuracy_comp1_tagged, iterations


def write_report_header(report_path, write_acc_test=False):
    with open(report_path, 'w') as report:
        report.write('model_index,threshold,fix_threshold,lam,maxiter,iterations,fix_weights1,' +
                     'fix_weights2,fix_weights3,fix_weights4,acc_comp')
        if write_acc_test:
            report.write(',acc_test')
        report.write('\n')


def write_report_line(report_path, model_index, threshold, fix_threshold, lam, maxiter, iterations, fix_weights,
                      acc_comp, acc_test=False):
    with open(report_path, 'a') as report:
        report.write(str(model_index) + ',')
        report.write(str(threshold) + ',')
        report.write(str(fix_threshold) + ',')
        report.write(str(lam) + ',')
        report.write(str(maxiter) + ',')
        report.write(str(iterations) + ',')
        report.write(str(fix_weights[0]) + ',')
        report.write(str(fix_weights[1]) + ',')
        report.write(str(fix_weights[2]) + ',')
        report.write(str(fix_weights[3]) + ',')
        report.write(str(acc_comp))
        if acc_test:
            report.write(',' + str(acc_test))
        report.write('\n')


if __name__ == '__main__':
    train = 'data/train1.wtag'
    report = 'dumps/report_small_11-05-2020.csv'
    write_report_header(report, write_acc_test=True)
    cv_small1(train, report)


    # model3: Log_Linear_MEMM = Log_Linear_MEMM.load_model('dumps/model3.pkl')  # train1.wtag accuracy 0.8591306489348602
    # model109: Log_Linear_MEMM = Log_Linear_MEMM.load_model('dumps/model109.pkl')  # train1.wtag accuracy 0.939752903993761
    #
    # with open('dumps/model109_comp1_predictions.pkl', 'rb') as f:
    #     predictions109 = pickle.load(f)
    #
    # model109.confusion_matrix('data/comp1_nltk_tagged.wtag', predictions109)
