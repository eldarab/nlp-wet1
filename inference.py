from math import exp
from auxiliary_functions import multiply_sparse, BEGIN, STOP, get_words_arr, exp_multiply_sparse
import numpy as np


def calc_q(feature_ids, weights, all_tags, pword, cword, nword, pptag, ptag, ctag):
    exp_weights = np.exp(weights)
    history = (cword, pptag, ptag, pword, nword)
    feature_rep = feature_ids.history_feature_representation(history, ctag)
    numerator = exp(multiply_sparse(weights, feature_rep))
    denominator = 0
    for tag in all_tags:
        feature_rep = feature_ids.history_feature_representation(history, tag)
        denominator += (exp_multiply_sparse(exp_weights, feature_rep))

    return numerator / denominator


def memm_viterbi(feature_ids, weights, sentence, beam_size):
    all_tags = feature_ids.get_all_tags()
    words_arr = [BEGIN] + get_words_arr(sentence) + [STOP]
    # Offsetting the size of the list to match the mathematical algorithm
    n = len(words_arr) - 2

    pi = [{} for i in range(n + 1)]
    bp = [{} for i in range(n + 1)]
    pi[0][(BEGIN, BEGIN)] = 1

    tags_dict = {-1: [BEGIN], 0: [BEGIN]}
    if beam_size == 0:
        tags_dict.update(dict.fromkeys([i for i in range(1, n+1)], all_tags))

    cword, nword = words_arr[0], words_arr[1]

    for k in range(1, n + 1):
        pword = cword
        cword = nword
        nword = words_arr[k + 1]

        beam_list = []
        for v in all_tags:
            v_prob = 0
            for u in tags_dict[k-1]:
                pi[k][u, v] = 0
                for t in tags_dict[k-2]:
                    if pi[k-1][t, u] == 0:
                        continue

                    q = calc_q(feature_ids, weights, all_tags, pword, cword, nword, t, u, v)
                    if pi[k-1][t, u] * q > pi[k][u, v]:
                        pi[k][u, v] = pi[k-1][t, u] * q
                        bp[k][u, v] = t
                v_prob += pi[k][u, v]
            beam_list.append((v, v_prob))
        beam_list.sort(reverse=True, key=lambda item: item[1])
        if beam_size != 0:
            tags_dict[k] = [beam_list[i][0] for i in range(beam_size)]

    tag_sequence = [None for i in range(n + 1)]
    max_prob = 0
    for u, v in pi[n].keys():
        if pi[n][(u, v)] > max_prob:
            max_prob = pi[n][(u, v)]
            tag_sequence[n - 1], tag_sequence[n] = u, v

    for k in range(n - 2, 0, -1):
        tag_sequence[k] = bp[k + 2][(tag_sequence[k + 1], tag_sequence[k + 2])]

    return tag_sequence[1:]
