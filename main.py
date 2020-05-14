from validation import *
from sys import argv
from time import time
from log_linear_memm import Log_Linear_MEMM

# TODO do not lehagish
if __name__ == '__main__':
    train_data = r'data\train1.wtag'
    test_data = r'data\test1.wtag'
    threshold, lam, maxiter = 100, 10, 100
    weights = (0.3, 0.5, 1, 1)

    # start_time = time()
    # model = Log_Linear_MEMM(threshold=threshold, lam=lam, maxiter=maxiter, fix_weights=weights)
    # model.fit(train_data)
    # train_time = time()
    # print('Finished optimizing, runtime', train_time-start_time)
    #
    # predictions = model.predict(test_data)
    # prediction_time = time()
    # print('Finished predicting, runtime', prediction_time-train_time)
    # print('Accuracy', Log_Linear_MEMM.accuracy(test_path=test_data, predictions=predictions))
    # Accuracy of 0.84278

    model = Log_Linear_MEMM.load_model(r'dumps\tested\train1 lambda-10 threshold-100 weights-(0.3, 0.5, 1, 1).pkl')
    predictions = model.predict(test_data)
    print('Accuracy', Log_Linear_MEMM.accuracy(test_path=test_data, predictions=predictions))
