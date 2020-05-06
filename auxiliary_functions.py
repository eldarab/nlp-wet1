import numpy as np

BEGIN = '*B'
STOP = '*S'
CONTAINS_DIGIT = '*CD'
CONTAINS_UPPER = '*CU'
CONTAINS_HYPHEN = '*CH'

# TODO check what functions is being used and is necessary


def multiply_sparse(v, f):
    res = 0
    for i in f:
        res += v[i]
    return res


def exp_multiply_sparse(v, f):
    res = 1
    for i in f:
        res *= v[i]
    return res


def add_or_append(dictionary, item, size=1):
    if item not in dictionary:
        dictionary[item] = size
    else:
        dictionary[item] += size


def parse_lower(word_tag):
    word, tag = word_tag.split('_')
    return word.lower(), tag


def get_words_arr(line):
    words_tags_arr = line.split(' ')
    if len(words_tags_arr) == 0:
        raise Exception("get_words_arr got an empty sentence.")
    if words_tags_arr[-1][-1:] == '\n':
        words_tags_arr[-1] = words_tags_arr[-1][:-1]  # removing \n from end of line
    return words_tags_arr


def get_line_tags(line):
    words_tags_arr = get_words_arr(line)
    tags = []
    for word_tag in words_tags_arr:
        if word_tag == '':
            continue
        tag = word_tag.split('_')[1]
        tags.append(tag)
    return tags


def get_file_tags(file):
    with open(file, 'r') as file:
        labels = []
        for line in file:
            labels = labels + get_line_tags(line)
    return labels


def has_digit(word):
    for char in word:
        if char.isdigit():
            return True


def has_upper(word):
    return not word.islower()


def has_hyphen(word):
    for char in word:
        if char == '-':
            return True


def sparse_to_dense(sparse_vec, dim):
    dense_vec = np.zeros(dim)
    for entrance in sparse_vec:
        dense_vec[entrance] += 1
    return dense_vec


# TODO check usages
def sparse_dict_to_dense(sparse_dict, dim):
    dense_vec = np.zeros(dim)
    for entrance in sparse_dict:
        dense_vec[entrance] = sparse_dict[entrance]
    return dense_vec


def get_all_histories_and_corresponding_tags(file_path):
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
                if i + 1 == len(words_tags_split):
                    nword = STOP
                else:
                    nword = words_tags_split[i + 1][0]
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


def clean_tags(input_data, file_name=None):
    with open(input_data, 'r') as in_file:
        if file_name is None:
            file_name = input_data[:-5] + '_clean.words'

        with open(file_name, 'w') as out_file:
            for line in in_file:
                words_tags = line.split()
                for word_tag in words_tags:
                    word = word_tag.split('_')[0]
                    out_file.write(word + ' ')
                out_file.write('\n')


def get_predictions_list(predictions):
    predicted_tags = []
    for sentence in predictions:
        for word_tag_tuple in sentence:
            predicted_tags.append(word_tag_tuple[1])
    return predicted_tags
