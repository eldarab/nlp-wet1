from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
    train_data = 'data/debugging_dataset.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)
    model.preprocess(threshold=10)
    model.optimize(lam=1, maxiter=100)
    # model.fit(train_path=train_data, threshold=10, lam=1)
    prediction = model.predict("Hadar went to the mall and bought some eggs .")
    print(prediction)
