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
    train_data = 'data/train1.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)
    model.preprocess(threshold=0)
    model.optimize(lam=0, maxiter=100, weights_path='dumps/weights_24-04-2020.pkl')
    prediction = model.predict('The Treasury is still working out the details with bank trade associations and the other government agencies that have a hand in fighting money laundering .')
    print(prediction)
