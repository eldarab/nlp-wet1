from log_linear_memm import Log_Linear_MEMM
from time import strftime


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
    time = strftime("%Y-%m-%d---%H-%M")
    train_data = 'data/debugging_dataset_1.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)
    model.preprocess(threshold=10, f100=True, f101=False, f102=False, f103=True, f104=True, f105=True,
                     f108=False, f109=False, f110=False)
    model.optimize(lam=1, maxiter=1, weights_path='dumps/weights_' + time + '.pkl')
    # prediction = model.predict('The Treasury is still working out the details with bank trade associations and the other government agencies that have a hand in fighting money laundering .')
    prediction = model.predict('The Treasury is still working the 64-day .')
    print(prediction)

