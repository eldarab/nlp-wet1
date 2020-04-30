import numpy as np
from re import search
from tqdm import tqdm

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
        raise Exception("get_words_arr got an empty sentence.")
    if words_tags_arr[-1][-2:] == '\n':
        words_tags_arr[-1] = words_tags_arr[-1][:-1]  # removing \n from end of line
    return words_tags_arr


def get_line_tags(line):
    words_tags_arr = get_words_arr(line)
    tags = []
    for word_tag in words_tags_arr:
        tag = word_tag.split('_')[1]
        tags.append(tag)
    return tags


def get_all_labels(file):
    with open(file, 'r') as file:
        labels = []
        for line in file:
            labels = labels + get_line_tags(line)
    return labels


# TODO check which implementation is faster
def has_digit(word):
    for char in word:
        if char.isdigit():
            return True
    # return any(char.isdigit() for char in word)
    # return bool(search(r'\d', word))


# TODO check which implementation is faster
def has_hyphen(word):
    for char in word:
        if char == '-':
            return True
    # return any(char == '_' for char in word)
    # return bool(search('-', word))


# TODO consider changing this to receive a master feature index
# def represent_history_with_features(feature_ids, sentence, i, pptag, ptag, ctag):
def represent_history_with_features(feature_ids, history, ctag):
    pword, cword, nword = history[4].lower(), history[0].lower(), history[3].lower()
    pptag, ptag = history[1], history[2]
    features = []
    has_upper = not history[0].islower()

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

    if has_upper and (CONTAINS_UPPER, ctag) in feature_ids.f109_index_dict:
        features.append(feature_ids.f109_index_dict[(CONTAINS_UPPER, ctag)])

    if has_hyphen(cword) and (CONTAINS_HYPHEN, ctag) in feature_ids.f110_index_dict:
        features.append(feature_ids.f110_index_dict[(CONTAINS_HYPHEN, ctag)])

    return np.array(features)


# This function does the same thing as the function above, only it returns a dense numpy array
def nd_history_feature_representation(feature_ids, history, ctag):
    pword, cword, nword = history[4].lower(), history[0].lower(), history[3].lower()
    pptag, ptag = history[1], history[2]
    has_upper = not history[0].islower()
    features_index = np.zeros(feature_ids.total_features)

    if (cword, ctag) in feature_ids.f100_index_dict:
        features_index[feature_ids.f100_index_dict[(cword, ctag)]] = 1

    for n in range(1, 5):
        if len(cword) <= n:
            break
        if (cword[:n], ctag) in feature_ids.f101_index_dict:
            features_index[feature_ids.f101_index_dict[(cword[:n], ctag)]] = 1
        if (cword[-n:], ctag) in feature_ids.f102_index_dict:
            features_index[feature_ids.f102_index_dict[(cword[-n:], ctag)]] = 1

    if (pptag, ptag, ctag) in feature_ids.f103_index_dict:
        features_index[feature_ids.f103_index_dict[(pptag, ptag, ctag)]] = 1

    if (ptag, ctag) in feature_ids.f104_index_dict:
        features_index[feature_ids.f104_index_dict[(ptag, ctag)]] = 1

    if ctag in feature_ids.f105_index_dict:
        features_index[feature_ids.f105_index_dict[ctag]] = 1

    if has_digit(cword) and (CONTAINS_DIGIT, ctag) in feature_ids.f108_index_dict:
        features_index[feature_ids.f108_index_dict[(CONTAINS_DIGIT, ctag)]] = 1

    if has_upper and (CONTAINS_UPPER, ctag) in feature_ids.f109_index_dict:
        features_index[feature_ids.f109_index_dict[(CONTAINS_UPPER, ctag)]] = 1

    if has_hyphen(cword) and (CONTAINS_HYPHEN, ctag) in feature_ids.f110_index_dict:
        features_index[feature_ids.f110_index_dict[(CONTAINS_HYPHEN, ctag)]] = 1

    return features_index


def calc_features_list(feature_ids, histories_list, ctags_list):
    return [represent_history_with_features(feature_ids, histories_list[i], ctags_list[i])
            for i in range(len(histories_list))]


def build_features_mat(feature_ids, all_histories_list, all_tags_list):
    row_dim = len(all_histories_list)
    col_dim = len(all_tags_list)
    feature_mat = [[represent_history_with_features(feature_ids, all_histories_list[i], all_tags_list[j])
                    for j in range(col_dim)] for i in range(row_dim)]
    return feature_mat


def calc_empirical_counts(features_list, dim):
    empirical_counts = np.zeros(dim)
    for feature in features_list:
        empirical_counts += sparse_to_dense(feature, dim)
    return empirical_counts


def calc_expected_counts_vec(features_matrix, dim):
    n = len(features_matrix)
    expected_counts_mat = np.empty((n, dim))
    i = 0  # row index, not using zip for efficiency
    for history in features_matrix:
        row = np.zeros(dim)
        for feature in history:
            row += sparse_to_dense(feature, dim)
        expected_counts_mat[i] = row
        i += 1
    return expected_counts_mat


def sparse_to_dense(sparse_vec, dim):
    dense_vec = np.zeros(dim)
    for entrance in sparse_vec:
        dense_vec[entrance] += 1
    return dense_vec


def get_all_histories_ctags(file_path):
    with open(file_path) as f:
        all_histories = []
        all_ctags = []
        for line in f:
            words_tags_arr = get_words_arr(line)
            if len(words_tags_arr) == 0:
                continue
            words_tags_split = [word_tag.split('_') for word_tag in words_tags_arr]
            ptag, ctag = BEGIN, BEGIN
            cword, nword = BEGIN, words_tags_split[0][0]
            for i in range(len(words_tags_split)):
                pptag = ptag
                ptag = ctag
                ctag = words_tags_split[i][1]
                pword = cword
                cword = words_tags_split[i][0]
                if i+1 == len(words_tags_split):
                    nword = STOP
                else:
                    nword = words_tags_split[i+1][0]
                history = (cword, pptag, ptag, pword, nword)
                all_histories.append(history)
                all_ctags.append(ctag)
    return all_histories, all_ctags


"""
def get_all_features_list(feature_ids, all_histories_list, all_ctags_list):
    all_features_list = []
    for history, ctag in all_histories_list, all_ctags_list:
        all_features_list.append(represent_history_with_features(feature_ids, history, ctag))
    return all_features_list
"""


def get_all_tags(file_path):
    tags = []
    with open(file_path, 'r') as f:
        for line in f:
            for word_tag in line.split():
                tag = word_tag.split('_')[1]
                if tag not in tags:
                    tags.append(tag)
    return tags
