from log_linear_memm import Log_Linear_MEMM
from metodot_ezer import *

if __name__ == '__main__':
    model = Log_Linear_MEMM(threshold=10, lam=0, maxiter=1, f101=False, f102=False)
    model.fit('data/debugging_dataset_200.wtag')
    model.save()
    model = load_model('dumps/test1.pkl')
    test_path = 'data/debugging_dataset_201_210.wtag'
    predictions = model.predict(test_path, beam_size=1)
    accuracy = Log_Linear_MEMM.accuracy(test_path, predictions)
    print(accuracy)
    Log_Linear_MEMM.confusion_matrix(test_path, predictions)
