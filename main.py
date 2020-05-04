from time import time
from auxiliary_functions import load_model
from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
    train_data = 'data/train1.wtag'
    test_data = 'data/debugging_dataset_201_210.wtag'
    start_time = time()

    model: Log_Linear_MEMM = load_model(r'dumps\model_2020-05-03_11-20-13.pkl')
    model.predict(r'data\debugging_dataset_20.wtag', beam_size=2)
    pass
