from validation import *
from sys import argv
from time import time
from log_linear_memm import Log_Linear_MEMM

# f102_dict = model.feature_statistics.f102_count_dict
# f102_dict_sorted = {k: v for k, v in sorted(f102_dict.items(), key=lambda item: item[1], reverse=True)}
#
# model3: Log_Linear_MEMM = Log_Linear_MEMM.load_model('dumps/model3.pkl')  # train1.wtag accuracy 0.8591306489348602
# model109: Log_Linear_MEMM = Log_Linear_MEMM.load_model('dumps/model109.pkl')  # train1.wtag accuracy 0.939752903993761
#
# with open('dumps/model109_comp1_predictions.pkl', 'rb') as f:
#     predictions109 = pickle.load(f)
#
# model109.confusion_matrix('data/comp1_nltk_tagged.wtag', predictions109)


if __name__ == '__main__':
    mode = argv[1]  # this will be used to run main simultaneously in different tmux sessions
    if mode == 'small1':
        print('small1')
        train_path = 'data/train2.wtag'
        report_path = 'dumps/report_small_10.csv'
        write_report_header(report_path, small_model=True)
        start_index = 620
        thresholds = [1]
        fix_thresholds = [1]
        lambdas = [0.45]
        fix_weights_list = [(1, 0.55, 1, 1), (1, 0.1, 1, 1), (1, 0.055, 1, 1), (1, 0.01, 1, 1), (1, 0.0055, 1, 1),
                            (1, 0.0001, 1, 1)]
        maxiter = 500
        validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list,
                 small_model=True)
    if mode == 'small2':
        print('small2')
        train_path = 'data/train2.wtag'
        report_path = 'dumps/report_small_11.csv'
        write_report_header(report_path, small_model=True)
        start_index = 630
        thresholds = [1]
        fix_thresholds = [1]
        lambdas = [0.45]
        fix_weights_list = [(1, 1, 1, 1)]
        maxiter = 500
        validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list,
                 small_model=True)
    if mode == 'train2acc':
        model602 = Log_Linear_MEMM.load_model('dumps/model602.pkl')
        predictions602 = model602.predict('data/train2.wtag', beam_size=2)
        accuracy602 = Log_Linear_MEMM.accuracy('data/train2.wtag', predictions602)
        print(accuracy602)
    if mode == 'big1':
        print('big1')
        train_path = 'data/train1.wtag'
        report_path = 'dumps/report_big_1.csv'
        write_report_header(report_path, small_model=False)
        start_index = 700
        thresholds = [1, 2]
        fix_thresholds = [2]
        lambdas = [0.1, 0.3, 0.5, 0.7, 1, 2, 3]
        fix_weights_list = [(1, 1, 1, 1)]
        maxiter = 500
        validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list,
                 small_model=False)


    train_data = r'data\train1.wtag'
    threshold, lam, maxiter = 50, 5, 100
    start_time = time()

    model = Log_Linear_MEMM(threshold=threshold, lam=lam, maxiter=maxiter).fit(train_data, iprint=1)
    model.save(filename="train_1_using_sparse_multiplication")

    test_path = r'data\debugging_dataset_200.wtag'
    model: Log_Linear_MEMM = Log_Linear_MEMM.load_model(r'dumps\train_1_using_sparse_multiplication.pkl')
    predictions = model.predict(test_path, beam_size=2)
    print(time()-start_time)
    print(Log_Linear_MEMM.accuracy(test_path, predictions))
    pass
