from time import time
from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
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
