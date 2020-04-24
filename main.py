from log_linear_memm import Log_Linear_MEMM


def clean_predictions(input_data):
    with open(input_data, 'r') as in_file:
        with open(input_data[:-5] + '_clean.txt', 'w') as out_file:
            for line in in_file:
                words_tags = line.split()
                for word_tag in words_tags:
                    word = word_tag.split('_')[0]
                    out_file.write(word + ' ')
                out_file.write('\n')


if __name__ == '__main__':
    train_data = 'data/debugging_dataset_2.wtag'
    clean_predictions(train_data)
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)
    model.preprocess(threshold=0)
    model.optimize(lam=1, maxiter=100)

    # model.load_weights('dumps/weights.pkl')

    prediction = model.predict('The Treasury is still working out the details with bank trade associations and the other government agencies that have a hand in fighting money laundering .')
    print(prediction)
