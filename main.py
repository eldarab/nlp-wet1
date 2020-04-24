from log_linear_memm import Log_Linear_MEMM


def clean_predictions(input_data):
    with open(input_data, 'r') as in_file:
        with open(input_data[:-5] + '_predictions.txt', 'w') as out_file:
            for line in in_file:
                words_tags = line.split()
                for word_tag in words_tags:
                    word = word_tag.split('_')[0]
                    out_file.write(word + ' ')


if __name__ == '__main__':
    train_data = 'data/train1.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)
    model.preprocess(threshold=10)
    model.optimize(lam=1, maxiter=100)
    # model.fit(train_path=train_data, threshold=10, lam=1)
    # prediction = model.predict("Hadar went to the mall and bought some eggs .")
    prediction = model.predict(train_data)
    print(prediction)
