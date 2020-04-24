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


# TODO change this to receive a master feature index
# def represent_history_with_features(feature_ids, sentence, i, pptag, ptag, ctag):
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

    # if not cword.islower() and (CONTAINS_UPPER, ctag) in feature_ids.f109_index_dict:
    if not cword.islower() and (CONTAINS_UPPER, ctag) in feature_ids.f109_index_dict:
        features.append(feature_ids.f109_index_dict[(CONTAINS_UPPER, ctag)])

    if has_hyphen(cword) and (CONTAINS_HYPHEN, ctag) in feature_ids.f110_index_dict:
        features.append(feature_ids.f110_index_dict[(CONTAINS_HYPHEN, ctag)])

    return features


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


def get_all_features_list(feature_ids, all_histories_list, all_ctags_list):
    all_features_list = []
    for history, ctag in all_histories_list, all_ctags_list:
        all_features_list.append(represent_history_with_features(feature_ids, history, ctag))
    return all_features_list


def get_all_tags(file_path):
    tags = []
    with open(file_path, 'r') as f:
        for line in f:
            for word_tag in line.split():
                tag = word_tag.split('_')[1]
                if tag not in tags:
                    tags.append(tag)
    return tags
