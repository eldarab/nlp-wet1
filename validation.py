from log_linear_memm import Log_Linear_MEMM
import pickle


# TODO old stuff
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
                print('model' + str(model_index) + ' finished training with test accuracy=' + str(accuracy_test))


def train_best(train_path, report_path):
    model_index = 200
    accuracy_test, accuracy_comp, iterations = evaluate(train_path, model_index, threshold=5, fix_threshold=100,
                                                        lam=0.5, maxiter=200, fix_weights=(1, 1, 1, 1))
    write_report_line(report_path, model_index, threshold=5, fix_threshold=100, lam=0.5, maxiter=200,
                      iterations=iterations, fix_weights=(1, 1, 1, 1), acc_comp=accuracy_comp, acc_test=accuracy_test)
    print('model' + str(model_index) + ' finished training with test accuracy=' + str(accuracy_test))
    model_index += 1
    accuracy_test, accuracy_comp, iterations = evaluate(train_path, model_index, threshold=5, fix_threshold=100,
                                                        lam=0.5, maxiter=100, fix_weights=(1, 1, 1, 1))
    write_report_line(report_path, model_index, threshold=5, fix_threshold=100, lam=0.5, maxiter=100,
                      iterations=iterations, fix_weights=(1, 1, 1, 1), acc_comp=accuracy_comp, acc_test=accuracy_test)
    print('model' + str(model_index) + ' finished training with test accuracy=' + str(accuracy_test))


def cv_small(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list):
    model_index = start_index
    for threshold in thresholds:
        for fix_threshold in fix_thresholds:
            for lam in lambdas:
                for fix_weights in fix_weights_list:
                    _, accuracy_comp, iterations = evaluate(train_path, model_index, threshold, fix_threshold, lam,
                                                            maxiter, fix_weights, small_model=True)
                    write_report_line(report_path, model_index, threshold, fix_threshold, lam, maxiter, iterations,
                                      fix_weights, acc_comp=accuracy_comp)
                    print('model' + str(model_index) + ' finished training with comp2 accuracy=' + str(accuracy_comp))
                    model_index += 1


def evaluate(train_path, model_index, threshold, fix_threshold, lam, maxiter, fix_weights, small_model=False):
    # creating and training a model
    model = Log_Linear_MEMM(threshold, fix_threshold, lam, maxiter, fix_weights)
    model.fit(train_path, iprint=-1)
    iterations = model.iter

    # saving trained model
    model.save('model' + str(model_index))

    # predicting and saving predictions
    if small_model:
        predictions_comp2 = model.predict('data/comp2_nltk_tagged.wtag', beam_size=2)
        with open('dumps/model' + str(model_index) + '_comp2_predictions.pkl', 'wb') as f:
            pickle.dump(predictions_comp2, f)
    else:
        predictions_test1 = model.predict('data/test1.wtag', beam_size=2)
        with open('dumps/model' + str(model_index) + '_test1_predictions.pkl', 'wb') as f:
            pickle.dump(predictions_test1, f)
        predictions_comp1 = model.predict('data/comp1_nltk_tagged.wtag', beam_size=2)
        with open('dumps/model' + str(model_index) + '_comp1_predictions.pkl', 'wb') as f:
            pickle.dump(predictions_comp1, f)

    # returning accuracy
    if small_model:
        accuracy_comp2_tagged = Log_Linear_MEMM.accuracy('data/comp2_nltk_tagged.wtag', predictions_comp2)
        return None, accuracy_comp2_tagged, iterations
    else:
        accuracy_test1 = Log_Linear_MEMM.accuracy('data/test1.wtag', predictions_test1)
        accuracy_comp1_tagged = Log_Linear_MEMM.accuracy('data/comp1_nltk_tagged.wtag', predictions_comp1)
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