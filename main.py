from log_linear_memm import Log_Linear_MEMM
from metodot_ezer import *

if __name__ == '__main__':
    model = Log_Linear_MEMM(threshold=10, lam=0, maxiter=1, f101=False, f102=False)
    model.fit('data/debugging_dataset_200.wtag')
    model.save('test1')
    model = load_model('dumps/test1.pkl')
    predictions = model.predict('data/debugging_dataset_201_210.wtag')
    print(predictions)
