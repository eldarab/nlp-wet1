from metodot_ezer import *
from math import exp
from preprocessing import *
import pickle

TRAIN_PATH = 'debugging_dataset.wtag'


def calc_q(feature_ids, weights, all_tags, pword, cword, nword, pptag, ptag, ctag):
    history = (cword, pptag, ptag, pword, nword)
    feature_rep = represent_history_with_features(feature_ids, history, ctag)
    numerator = exp(mult_sparse(weights, feature_rep))
    denominator = 0
    for tag in all_tags:
        feature_rep = represent_history_with_features(feature_ids, history, tag)
        denominator += exp(mult_sparse(weights, feature_rep))

    return numerator / denominator


def memm_viterbi(feature_ids, weights, all_tags, sentence):
    words_arr = get_words_arr(sentence)
    n = len(words_arr)

    pi = [{} for i in range(n+1)]
    bp = [{} for i in range(n+1)]
    pi[0][(BEGIN, BEGIN)] = 1

    cword, nword = BEGIN, BEGIN
    for k in range(1, n+1):
        pword = cword
        cword = nword
        if k < n-1:
            nword = words_arr[k+1]
        else:
            nword = STOP

        for v in all_tags:
            if k == 1:
                pi[1][(BEGIN, v)] = calc_q(feature_ids, weights, all_tags, pword, cword, nword, BEGIN, BEGIN, v)
                # No need for setting a value for bp because it is only used for k >= 3
            else:
                for u in all_tags:
                    if k == 2:
                        pi[2][(u, v)] = pi[1][(BEGIN, u)] * \
                                        calc_q(feature_ids, weights, all_tags, pword, cword, nword, BEGIN, u, v)
                        # Not setting value for bp
                    else:
                        pi[k][(u, v)] = 0
                        for t in all_tags:
                            q = calc_q(feature_ids, weights, all_tags, pword, cword, nword, t, u, v)
                            if pi[k-1][(t, u)] * q > pi[k][(u, v)]:
                                pi[k][(u, v)] = pi[k-1][(t, u)]
                                bp[k][(u, v)] = t

    tag_sequence = [None for i in range(n+1)]
    max_prob = 0
    for u in all_tags:
        for v in all_tags:
            if pi[n][(u, v)] > max_prob:
                max_prob = pi[n][(u, v)]
                tag_sequence[n-1], tag_sequence[n] = u, v

    for k in range(n-2, 0, -1):
        tag_sequence[k] = bp[k+2][(tag_sequence[k+1], tag_sequence[k+2])]

    return tag_sequence[1:]


if __name__ == '__main__':
    statistics = feature_statistics_class(TRAIN_PATH)
    feature2id = feature2id_class(TRAIN_PATH, statistics, threshold=10)

    with open("pickelim/trained_weights_data_i.pkl", 'rb') as f:
        optimal_params = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    test = "Hadar went to the mall and bought some eggs ."
    tags = memm_viterbi(feature2id, pre_trained_weights, feature2id.get_all_tags(), test)
    pass
