from log_linear_memm import Log_Linear_MEMM
from time import strftime
from emailer import send_email


# TODO do not lehagish
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
    # clean_predictions('data/debugging_dataset_201_210.wtag')
    start_time = strftime("%Y-%m-%d_%H-%M-%S")
    train_data = 'data/debugging_dataset_200.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)

    #   Preprocessing
    model.preprocess(threshold=10, f100=True, f101=False, f102=False, f103=True,
                     f104=True, f105=True, f108=False, f109=False, f110=False)
    preprocess_time = strftime("%Y-%m-%d_%H-%M-%S")

    #   Optimizing
    # model.optimize(lam=1, maxiter=25, weights_path='dumps/weights_' + start_time + '.pkl')
    optimization_time = strftime("%Y-%m-%d_%H-%M-%S")

    #   Load pre-trained weights
    model.load_weights('dumps/weights_2020-04-30_12-47-31.pkl')

    #   Predict
    # prediction = model.predict('data/debugging_dataset_201_210_clean.txt')
    prediction = model.predict('Michael , which also processes potatoes , still relies on spuds for about a fourth of its sales and nearly half its pretax profit . ')
    prediction_time = strftime("%Y-%m-%d_%H-%M-%S")

    #  End message
    message_body = 'Start: ' + start_time + '\nPreprocess end: ' + preprocess_time + '\nOptimization end: ' + \
                   optimization_time + '\nPrediction end: ' + prediction_time

    # send_email('eldar.abraham@gmail.com', '<my penis>', ['eldar.a@campus.technion.ac.il'], message_body)
    # model.load_weights('dumps/weights.pkl')
