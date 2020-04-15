from collections import OrderedDict

from metodot_ezer import *


class feature_statistics_class:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f100_count_dict = OrderedDict()  # Init all features dictionaries
        self.f101_count_dict = OrderedDict()  # Prefix features
        self.f102_count_dict = OrderedDict()  # Suffix features
        self.f103_count_dict = OrderedDict()  # Trigram features
        self.f104_count_dict = OrderedDict()  # Bigram features
        self.f105_count_dict = OrderedDict()  # Unigram features
        self.f108_count_dict = OrderedDict()  # Contain Number features
        self.f109_count_dict = OrderedDict()  # Contain Uppercase features
        self.f110_count_dict = OrderedDict()  # Contain Hyphen features
        self.count_f100()
        self.count_f101()
        self.count_f102()
        self.count_f103()
        self.count_f104()
        self.count_f105()
        self.count_f108()
        self.count_f109()
        self.count_f110()

    def count_f100(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = parse_lower(word_tag)
                    add_or_append(self.f100_count_dict, (cword, ctag))

    def count_f101(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = parse_lower(word_tag)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        add_or_append(self.f101_count_dict, (cword[:n], ctag))

    def count_f102(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = parse_lower(word_tag)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        add_or_append(self.f102_count_dict, (cword[-n:], ctag))

    def count_f103(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                ptag = BEGIN
                ctag = BEGIN
                for word_tag in words_tags_arr:
                    pptag = ptag
                    ptag = ctag
                    ctag = word_tag.split('_')[1]
                    add_or_append(self.f103_count_dict, (pptag, ptag, ctag))

    def count_f104(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                ctag = BEGIN
                for word_tag in words_tags_arr:
                    ptag = ctag
                    ctag = word_tag.split('_')[1]
                    add_or_append(self.f104_count_dict, (ptag, ctag))

    def count_f105(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    ctag = word_tag.split('_')[1]
                    add_or_append(self.f105_count_dict, ctag)

    def count_f108(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = word_tag.split('_')[0], word_tag.split('_')[1]
                    if has_digit(cword):
                        add_or_append(self.f108_count_dict, (CONTAINS_DIGIT, ctag))

    def count_f109(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = word_tag.split('_')
                    if not cword.islower():
                        add_or_append(self.f109_count_dict, (CONTAINS_UPPER, ctag))

    def count_f110(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = word_tag.split('_')[0], word_tag.split('_')[1]
                    if has_hyphen(cword):
                        add_or_append(self.f110_count_dict, (CONTAINS_HYPHEN, ctag))


class feature2id_class:
    def __init__(self, file_path, feature_statistics, threshold):
        self.file_path = file_path
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.total_features = 0  # Total number of features accumulated
        # Internal feature indexing
        self.f100_counter = 0
        self.f101_counter = 0
        self.f102_counter = 0
        self.f103_counter = 0
        self.f104_counter = 0
        self.f105_counter = 0
        self.f108_counter = 0
        self.f109_counter = 0
        self.f110_counter = 0
        # Init all features dictionaries
        self.f100_index_dict = OrderedDict()
        self.f101_index_dict = OrderedDict()
        self.f102_index_dict = OrderedDict()
        self.f103_index_dict = OrderedDict()
        self.f104_index_dict = OrderedDict()
        self.f105_index_dict = OrderedDict()
        self.f108_index_dict = OrderedDict()
        self.f109_index_dict = OrderedDict()
        self.f110_index_dict = OrderedDict()
        self.initialize_f100_index_dict()
        self.initialize_f101_index_dict()
        self.initialize_f102_index_dict()
        self.initialize_f103_index_dict()
        self.initialize_f104_index_dict()
        self.initialize_f105_index_dict()
        self.initialize_f108_index_dict()
        self.initialize_f109_index_dict()
        self.initialize_f110_index_dict()

    def get_all_tags(self):
        return [tag for tag in self.f105_index_dict.keys()]

    # TODO create this function and put it into use
    def get_master_index(self):
        pass

    def initialize_f100_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = parse_lower(word_tag)
                    # TODO create a function that performs this:
                    if (cword, ctag) not in self.f100_index_dict \
                            and self.feature_statistics.f100_count_dict[(cword, ctag)] >= self.threshold:
                        self.f100_index_dict[(cword, ctag)] = self.f100_counter + self.total_features
                        # TODO compare to colab example
                        self.f100_counter += 1
        self.total_features += self.f100_counter

    def initialize_f101_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = parse_lower(word_tag)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        prefix = cword[:n]
                        if (prefix, ctag) not in self.f101_index_dict \
                                and self.feature_statistics.f101_count_dict[(prefix, ctag)] >= self.threshold:
                            self.f101_index_dict[(prefix, ctag)] = self.f101_counter + self.total_features
                            self.f101_counter += 1
        self.total_features += self.f101_counter

    def initialize_f102_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = parse_lower(word_tag)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        suffix = cword[-n:]
                        if (suffix, ctag) not in self.f102_index_dict \
                                and self.feature_statistics.f102_count_dict[(suffix, ctag)] >= self.threshold:
                            self.f102_index_dict[(suffix, ctag)] = self.f102_counter + self.total_features
                            self.f102_counter += 1
        self.total_features += self.f102_counter

    def initialize_f103_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                ptag = BEGIN
                ctag = BEGIN
                for word_tag in words_tags_arr:
                    pptag = ptag
                    ptag = ctag
                    ctag = word_tag.split('_')[1]
                    if (pptag, ptag, ctag) not in self.f103_index_dict \
                            and self.feature_statistics.f103_count_dict[(pptag, ptag, ctag)] >= self.threshold:
                        self.f103_index_dict[(pptag, ptag, ctag)] = self.f103_counter + self.total_features
                        self.f103_counter += 1
        self.total_features += self.f103_counter

    def initialize_f104_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                ctag = BEGIN
                for word_tag in words_tags_arr:
                    ptag = ctag
                    ctag = word_tag.split('_')[1]
                    if (ptag, ctag) not in self.f104_index_dict \
                            and self.feature_statistics.f104_count_dict[(ptag, ctag)] >= self.threshold:
                        self.f104_index_dict[(ptag, ctag)] = self.f104_counter + self.total_features
                        self.f104_counter += 1
        self.total_features += self.f104_counter

    def initialize_f105_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    ctag = word_tag.split('_')[1]
                    if ctag not in self.f105_index_dict \
                            and self.feature_statistics.f105_count_dict[ctag] >= self.threshold:
                        self.f105_index_dict[ctag] = self.f105_counter + self.total_features
                        self.f105_counter += 1
        self.total_features += self.f105_counter

    def initialize_f108_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = word_tag.split('_')[0], word_tag.split('_')[1]
                    if has_digit(cword):
                        if (CONTAINS_DIGIT, ctag) not in self.f108_index_dict \
                                and self.feature_statistics.f108_count_dict[(CONTAINS_DIGIT, ctag)] >= self.threshold:
                            self.f108_index_dict[(CONTAINS_DIGIT, ctag)] = self.f108_counter + self.total_features
                            self.f108_counter += 1
        self.total_features += self.f108_counter

    def initialize_f109_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = word_tag.split('_')
                    if not cword.islower():
                        pair = (CONTAINS_UPPER, ctag)
                        if pair not in self.f109_index_dict \
                                and self.feature_statistics.f109_count_dict[pair] >= self.threshold:
                            self.f109_index_dict[pair] = self.f109_counter + self.total_features
                            self.f109_counter += 1
        self.total_features += self.f109_counter

    def initialize_f110_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = word_tag.split('_')
                    if has_hyphen(cword):
                        if (CONTAINS_HYPHEN, ctag) not in self.f110_index_dict \
                                and self.feature_statistics.f110_count_dict[(CONTAINS_HYPHEN, ctag)] >= self.threshold:
                            self.f110_index_dict[(CONTAINS_HYPHEN, ctag)] = self.f110_counter + self.total_features
                            self.f110_counter += 1
        self.total_features += self.f110_counter


if __name__ == '__main__':
    file_path = 'data/train1.wtag'
    stats = feature_statistics_class(file_path)
    aids = feature2id_class(file_path, stats, 4)
    history1 = ('went', '*B', 'NN', 'VBD', 'Eldar', 'to')
    pass
