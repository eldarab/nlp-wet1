from time import time

from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
    start_time = time()
    model = Log_Linear_MEMM(threshold=10, lam=5, maxiter=50)
    model.fit('data/debugging_dataset_200.wtag')
    fit_time = time()
    # predictions = model.predict('data/debugging_dataset_201_210.wtag', beam_size=1)
    predictions_time = time()
    print("Model fit time", str(fit_time-start_time), "\nPrediction time", str(predictions_time-fit_time))
    # LogLinearMEMM.create_predictions_file(predictions)
    pass
