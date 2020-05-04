from time import time
from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
    train_data = 'data/train1.wtag'
    test_data = r'data\debugging_dataset_20.wtag'
    start_time = time()

    model = Log_Linear_MEMM.load_model(r'dumps\model_2020-05-03_11-20-13.pkl')
    predictions = model.predict(test_data, beam_size=2)
    print(Log_Linear_MEMM.accuracy(test_data, predictions))
    pass
