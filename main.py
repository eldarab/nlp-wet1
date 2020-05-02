from log_linear_memm import Log_Linear_MEMM
from metodot_ezer import *

if __name__ == '__main__':
    model1 = load_model('dumps/model1.pkl')
    grad1 = model1.lbfgs_result[2]['grad']
    print('grad1: ' + str(np.linalg.norm(grad1)))

    model2 = load_model('dumps/model2.pkl')
    grad2 = model2.lbfgs_result[2]['grad']
    print('grad2: ' + str(np.linalg.norm(grad2)))

    model3 = load_model('dumps/model3.pkl')
    grad3 = model3.lbfgs_result[2]['grad']
    print('grad3: ' + str(np.linalg.norm(grad3)))

    model4 = load_model('dumps/model4.pkl')
    grad4 = model4.lbfgs_result[2]['grad']
    print('grad4: ' + str(np.linalg.norm(grad4)))

    predictions1 = model1.predict('data/debugging_dataset_200.wtag', 1)
    accuracy1 = Log_Linear_MEMM().accuracy('data/debugging_dataset_200.wtag', predictions1)
    print('accuracy of model 1: ' + str(accuracy1))

    predictions2 = model2.predict('data/debugging_dataset_200.wtag', 1)
    accuracy2 = Log_Linear_MEMM().accuracy('data/debugging_dataset_200.wtag', predictions2)
    print('accuracy of model 2: ' + str(accuracy2))

    predictions3 = model3.predict('data/debugging_dataset_200.wtag', 1)
    accuracy3 = Log_Linear_MEMM().accuracy('data/debugging_dataset_200.wtag', predictions3)
    print('accuracy of model 3: ' + str(accuracy3))

    predictions4 = model4.predict('data/debugging_dataset_200.wtag', 1)
    accuracy4 = Log_Linear_MEMM().accuracy('data/debugging_dataset_200.wtag', predictions4)
    print('accuracy of model 4: ' + str(accuracy4))
