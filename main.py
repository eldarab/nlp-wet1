from validation import *
from sys import argv

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
        fix_weights_list = [(0.55, 1, 1, 1), (0.1, 1, 1, 1), (0.055, 1, 1, 1), (0.01, 1, 1, 1), (0.0055, 1, 1, 1),
                            (0.0001, 1, 1, 1)]
        maxiter = 500
        validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list,
                 small_model=True)
    if mode == 'train2acc':
        print('hadargay')

