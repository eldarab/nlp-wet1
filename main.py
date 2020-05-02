from log_linear_memm import Log_Linear_MEMM
from auxiliary_functions import *


def evaluate(train_path, model_index, threshold, fix_threshold, lam):
    # creating and training a model
    model = Log_Linear_MEMM(threshold, fix_threshold, lam)
    model.fit(train_path)  # using default maxiter=200

    # saving trained model
    model.save('model' + str(model_index))

    # predicting and saving predictions
    predictions_test1 = model.predict('data/debugging_dataset_201_210.wtag', beam_size=2)  # TODO change test1.wtag
    with open('dumps/model' + str(model_index) + '_test1_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_test1, f)
    predictions_comp1 = model.predict('data/debugging_dataset_201_210.wtag', beam_size=2)  # TODO change comp1_nltk_tagged.wtag
    with open('dumps/model' + str(model_index) + '_comp1_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_comp1, f)

    # returning accuracy on test1.wtag and comp1_nltk_tagged.wtag
    accuracy_test1 = Log_Linear_MEMM.accuracy('data/debugging_dataset_201_210.wtag', predictions_test1)  # TODO change test1.wtag
    accuracy_comp1_tagged = Log_Linear_MEMM.accuracy('data/debugging_dataset_201_210.wtag', predictions_comp1)  # TODO change comp1_nltk_tagged.wtag
    return accuracy_test1, accuracy_comp1_tagged


if __name__ == '__main__':
    train_path = 'data/debugging_dataset_20.wtag'  # TODO change train1.wtag
    model_index = 50  # starting from big number to make sure we dont confuse with previously trained models
    with open('dumps/report_02-05-2020.txt', 'w') as report:
        report.write('---CV Report 02-05-2020---\n\n')
    thresholds = [5, 10, 15, 20]
    fix_thresholds = [100, 200, 250]
    lambdas = [0.1, 0.5, 0.7]  # TODO choose better lambdas
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

    # model = load_model('dumps/model3.pkl')
    # f102_dict = model.feature_statistics.f102_count_dict
    # f102_dict_sorted = {k: v for k, v in sorted(f102_dict.items(), key=lambda item: item[1], reverse=True)}
    # pass





