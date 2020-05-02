from time import time

from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
    start_time = time()
    new_model = Log_Linear_MEMM(threshold=10, lam=5, maxiter=50)
    new_model.fit('data/debugging_dataset_200.wtag', True)
