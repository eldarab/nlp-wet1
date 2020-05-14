from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
    # TODO insert paths
    model_1_pickle_path = ''
    model_2_pickle_path = ''
    comp1_path = ''
    comp2_path = ''

    model_1: Log_Linear_MEMM = Log_Linear_MEMM.load_model(model_1_pickle_path)
    model_1_prediction = model_1.predict(comp1_path)
    Log_Linear_MEMM.create_predictions_file(model_1_prediction, file_name='comp_m1_318792827')

    model_2: Log_Linear_MEMM = Log_Linear_MEMM.load_model(model_2_pickle_path)
    model_2_prediction = model_2.predict(comp2_path)
    Log_Linear_MEMM.create_predictions_file(model_2_prediction, file_name='comp_m2_318792827')
