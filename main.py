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
    start_time = strftime("%Y-%m-%d_%H-%M-%S")
    train_data = 'data/debugging_dataset_1.wtag'
    model = Log_Linear_MEMM()
    model.set_train_path(train_data)
    model.preprocess(threshold=0, f100=True, f101=False, f102=False, f103=True,
                     f104=True, f105=True, f108=False, f109=False, f110=False)
    preprocess_time = strftime("%Y-%m-%d_%H-%M-%S")
    model.optimize(lam=0, maxiter=10, weights_path='dumps/weights_' + start_time + '.pkl')
    optimization_time = strftime("%Y-%m-%d_%H-%M-%S")
    # prediction = model.predict('The Treasury is still working out the details with bank trade associations \
    # and the other government agencies that have a hand in fighting money laundering .')
    prediction = model.predict('The Treasury is still working the .\n')
    print(prediction)
    prediction_time = strftime("%Y-%m-%d_%H-%M-%S")
    message_body = 'Start: ' + start_time + '\nPreprocess end: ' + preprocess_time + '\nOptimization end: ' + \
                   optimization_time + '\nPrediction end: ' + prediction_time
    # send_email('eldar.abraham@gmail.com', '<my penis>', ['eldar.a@campus.technion.ac.il'], message_body)
    # model.load_weights('dumps/weights.pkl')

    # prediction = model.predict('The Treasury is still working out the details with bank trade associations and the other government agencies that have a hand in fighting money laundering .')
    # print(prediction)
