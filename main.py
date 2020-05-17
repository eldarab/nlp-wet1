from validation import *
from sys import argv
from log_linear_memm import Log_Linear_MEMM


if __name__ == '__main__':
    mode = argv[1]
    if mode == 'big_final':
        print('big_final')
        train_path = 'data/train1.wtag'
        report_path = 'dumps/report_big_3_after_change.csv'
        write_report_header(report_path, small_model=False)
        start_index = 1000
        thresholds = [1]
        fix_thresholds = [2, 3]
        lambdas = [0.4]
        fix_weights_list = [(1, 1, 1, 1)]
        maxiter = 500
        validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list,
                 small_model=False)
    if mode == 'small_final':
        print('small_final')
        train_path = 'data/train2.wtag'
        report_path = 'dumps/report_small_2_after_change.csv'
        write_report_header(report_path, small_model=True)
        start_index = 2100
        thresholds = [2, 3, 4]
        fix_thresholds = [1, 2, 3, 4]
        lambdas = [0.3, 0.32, 0.34, 0.36, 0.38, 0.4]
        fix_weights_list = [(1, 1, 1, 1)]
        maxiter = 500
        validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter,
                 fix_weights_list,
                 small_model=True)
