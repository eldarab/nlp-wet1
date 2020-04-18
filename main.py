from log_linear_memm import Log_Linear_MEMM

if __name__ == '__main__':
    train_data = 'data/train1.wtag'
    model = Log_Linear_MEMM()
    model.fit(train_path=train_data, threshold=10, lam=1)
    prediction = model.predict("Hadar went to the mall and bought some eggs .")
    print(prediction)
