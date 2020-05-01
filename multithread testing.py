from log_linear_memm import Log_Linear_MEMM
from time import strftime, time


if __name__ == '__main__':
    start_time = strftime("%Y-%m-%d_%H-%M-%S")
    train_data = 'data/debugging_dataset_200.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)

    threshold = 10
    model.preprocess(threshold=threshold)
    preprocess_time = time()

    lam = 0
    maxiter = 50
    model.optimize(lam=lam, maxiter=maxiter, weights_path='dumps/weights_' + train_data[5:-5] + '_threshold=' +
                          str(threshold) + '_lam=' + str(lam) + '_iter=' + str(maxiter) + '_' + start_time + '.pkl')
    optimization_time = time()
    print(optimization_time - preprocess_time)
