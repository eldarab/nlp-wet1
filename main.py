from log_linear_memm import Log_Linear_MEMM
from metodot_ezer import *

if __name__ == '__main__':
    print('training model1')
    model1 = Log_Linear_MEMM(threshold=10, lam=0, maxiter=100, f101=False, f102=False)
    model1.fit('data/train1.wtag')
    model1.save('model1')

    print('training model2')
    model2 = Log_Linear_MEMM(threshold=10, lam=0, maxiter=100, fix_threshold=100)
    model2.fit('data/train1.wtag')
    model2.save('model2')

    print('training model3')
    model3 = Log_Linear_MEMM(threshold=10, lam=0.1, maxiter=100, fix_threshold=100)
    model3.fit('data/train1.wtag')
    model3.save('model3')

    print('training model4')
    model4 = Log_Linear_MEMM(threshold=10, lam=1, maxiter=100, fix_threshold=100)
    model4.fit('data/train1.wtag')
    model4.save('model4')

    print('training model5')
    model5 = Log_Linear_MEMM(threshold=10, lam=10, maxiter=100, fix_threshold=100)
    model5.fit('data/train1.wtag')
    model5.save('model5')

    print('training model6')
    model6 = Log_Linear_MEMM(threshold=10, lam=100, maxiter=100, fix_threshold=100)
    model6.fit('data/train1.wtag')
    model6.save('model6')
