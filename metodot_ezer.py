import numpy as np


BEGIN = '*B'
STOP = '*S'
CONTAINS_DIGIT = '*CD'
CONTAINS_UPPER = '*CU'
CONTAINS_HYPHEN = '*CH'


def mult_sparse(v, f):
    res = 0
    for i in f:
        res += v[i]
    return res


def add_or_append(dictionary, item):
    if item not in dictionary:
        dictionary[item] = 1
    else:
        dictionary[item] += 1


def parse_lower(word_tag):
    word, tag = word_tag.split('_')
    return word.lower(), tag


def get_words_arr(line):
    words_tags_arr = line.split(' ')
    if len(words_tags_arr) == 0:
        return
    words_tags_arr[-1] = words_tags_arr[-1][:-1]  # removing \n from end of line
    return words_tags_arr


def has_digit(word):
    for char in word:
        if char.isdigit():
            return True


def has_hyphen(word):
    for char in word:
        if char == '-':
            return True


def represent_history_with_features(feature_ids, history, ctag):
    pword, cword, nword = history[4], history[0], history[3]
    pptag, ptag = history[1], history[2]
    features = []

    if (cword, ctag) in feature_ids.f100_index_dict:
        features.append(feature_ids.f100_index_dict[(cword, ctag)])

    for n in range(1, 5):
        if len(cword) <= n:
            break
        if (cword[:n], ctag) in feature_ids.f101_index_dict:
            features.append(feature_ids.f101_index_dict[(cword[:n], ctag)])
        if (cword[-n:], ctag) in feature_ids.f102_index_dict:
            features.append(feature_ids.f102_index_dict[(cword[-n:], ctag)])

    if (pptag, ptag, ctag) in feature_ids.f103_index_dict:
        features.append(feature_ids.f103_index_dict[(pptag, ptag, ctag)])

    if (ptag, ctag) in feature_ids.f104_index_dict:
        features.append(feature_ids.f104_index_dict[(ptag, ctag)])

    if ctag in feature_ids.f105_index_dict:
        features.append(feature_ids.f105_index_dict[ctag])

    if has_digit(cword) and (CONTAINS_DIGIT, ctag) in feature_ids.f108_index_dict:
        features.append(feature_ids.f108_index_dict[(CONTAINS_DIGIT, ctag)])

    if not cword.lower() and (CONTAINS_UPPER, ctag) in feature_ids.f109_index_dict:
        features.append(feature_ids.f109_index_dict[(CONTAINS_UPPER, ctag)])

    if has_hyphen(cword) and (CONTAINS_HYPHEN, ctag) in feature_ids.f110_index_dict:
        features.append(feature_ids.f110_index_dict[(CONTAINS_HYPHEN, ctag)])

    return features


def calc_features_list(feature_ids, histories_list, ctags_list):
    return [represent_history_with_features(feature_ids, histories_list[i], ctags_list[i]) \
            for i in range(len(histories_list))]


def calc_empirical_counts(feature_ids, features_list):
    dim = feature_ids.total_features
    empirical_counts = np.zeros(dim)
    for feature in features_list:
        for i in feature:
            empirical_counts[i] += 1
    return empirical_counts
