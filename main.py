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


def evaluate(train_path, model_index, threshold, fix_threshold, lam):
    # creating and training a model
    model = Log_Linear_MEMM(threshold, fix_threshold, lam)
    model.fit(train_path)  # using default maxiter=200

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
    return accuracy_test1, accuracy_comp1_tagged


if __name__ == '__main__':
    model3 = Log_Linear_MEMM.load_model('dumps/model3.pkl')  # train1.wtag accuracy 0.8591306489348602
    model109 = Log_Linear_MEMM.load_model('dumps/model109.pkl')  # train1.wtag accuracy 0.939752903993761

    print(model3 - model109)
    pass
    # model_try = Log_Linear_MEMM(threshold=10, fix_threshold=10, lam=0.1)
    # model_try.fit('data/debugging_dataset_200.wtag')
    # predictions = model_try.predict('data/test1.wtag', beam_size=2)
    # accuracy = Log_Linear_MEMM.accuracy('data/test1.wtag', predictions)
    # print(accuracy)
