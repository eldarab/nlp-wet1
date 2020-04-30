from math import exp
from preprocessing import *

from tqdm import tqdm


def calc_q(feature_ids, weights, all_tags, pword, cword, nword, pptag, ptag, ctag):
    history = (cword, pptag, ptag, pword, nword)
    feature_rep = represent_history_with_features(feature_ids, history, ctag)
    numerator = exp(mult_sparse(weights, feature_rep))
    denominator = 0
    for tag in all_tags:
        feature_rep = represent_history_with_features(feature_ids, history, tag)
        denominator += exp(mult_sparse(weights, feature_rep))

    return numerator / denominator


# TODO implement beam-search
def memm_viterbi(feature_ids, weights, all_tags, sentence, beam_size=None):
    words_arr = [BEGIN] + get_words_arr(sentence) + [STOP]
    # Offsetting the size of the list to match the mathematical algorithm
    n = len(words_arr) - 2

    pi = [{} for i in range(n + 1)]
    bp = [{} for i in range(n + 1)]
    # TODO make this useful or delete it
    # pi[0][(BEGIN, BEGIN)] = 1

    cword, nword = words_arr[0], words_arr[1]

    for k in tqdm(range(1, n + 1)):
        pword = cword
        cword = nword
        nword = words_arr[k + 1]

        for v in all_tags:
            if k == 1:
                pi[1][(BEGIN, v)] = calc_q(feature_ids, weights, all_tags, pword, cword, nword, BEGIN, BEGIN, v)
                # No need for setting a value for bp because it is only used for k >= 3
                continue

            for u in all_tags:
                if k == 2:
                    q = calc_q(feature_ids, weights, all_tags, pword, cword, nword, BEGIN, u, v)
                    pi[2][(u, v)] = pi[1][(BEGIN, u)] * q
                    # Not setting value for bp
                    continue

                pi[k][(u, v)] = 0
                for t in all_tags:
                    q = calc_q(feature_ids, weights, all_tags, pword, cword, nword, t, u, v)
                    if pi[k - 1][(t, u)] * q > pi[k][(u, v)]:
                        pi[k][(u, v)] = pi[k - 1][(t, u)] * q
                        bp[k][(u, v)] = t

    tag_sequence = [None for i in range(n + 1)]
    max_prob = 0
    for u, v in pi[n].keys():
        if pi[n][(u, v)] > max_prob:
            max_prob = pi[n][(u, v)]
            tag_sequence[n - 1], tag_sequence[n] = u, v

    for k in range(n - 2, 0, -1):
        tag_sequence[k] = bp[k + 2][(tag_sequence[k + 1], tag_sequence[k + 2])]

    return tag_sequence[1:]
