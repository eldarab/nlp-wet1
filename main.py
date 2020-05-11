from validation import *


# f102_dict = model.feature_statistics.f102_count_dict
# f102_dict_sorted = {k: v for k, v in sorted(f102_dict.items(), key=lambda item: item[1], reverse=True)}


if __name__ == '__main__':
    train_path = 'data/train2.wtag'
    report_path = 'dumps/report_small_2.csv'
    write_report_header(report_path, small_model=False)
    start_index = 300
    thresholds = [5, 4, 3, 2, 1]
    fix_thresholds = [50, 30, 10, 5]
    lambdas = [10, 0.1, 0.01, 0]
    maxiter = 500
    fix_weights_list = [(1, 1, 1, 1)]
    validate(train_path, report_path, start_index, thresholds, fix_thresholds, lambdas, maxiter, fix_weights_list)



    # model3: Log_Linear_MEMM = Log_Linear_MEMM.load_model('dumps/model3.pkl')  # train1.wtag accuracy 0.8591306489348602
    # model109: Log_Linear_MEMM = Log_Linear_MEMM.load_model('dumps/model109.pkl')  # train1.wtag accuracy 0.939752903993761
    #
    # with open('dumps/model109_comp1_predictions.pkl', 'rb') as f:
    #     predictions109 = pickle.load(f)
    #
    # model109.confusion_matrix('data/comp1_nltk_tagged.wtag', predictions109)
