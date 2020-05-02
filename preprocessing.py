from collections import OrderedDict
from auxiliary_functions import *


class FeatureStatisticsClass:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f100_count_dict = OrderedDict()  # Init all features dictionaries
        self.f101_count_dict = OrderedDict()  # Prefix features
        self.f102_count_dict = OrderedDict()  # Suffix features
        self.f103_count_dict = OrderedDict()  # Trigram features
        self.f104_count_dict = OrderedDict()  # Bigram features
        self.f105_count_dict = OrderedDict()  # Unigram features
        self.f106_count_dict = OrderedDict()  # Previous word + tag
        self.f107_count_dict = OrderedDict()  # Next word + tag
        self.f108_count_dict = OrderedDict()  # Contain Number features
        self.f109_count_dict = OrderedDict()  # Contain Uppercase features
        self.f110_count_dict = OrderedDict()  # Contain Hyphen features

    def count_features(self, f100, f101, f102, f103, f104, f105, f106, f107, f108, f109, f110):
        if f100:
            self.count_f100()
        if f101:
            self.count_f101()
        if f102:
            self.count_f102()
        if f103:
            self.count_f103()
        if f104:
            self.count_f104()
        if f105:
            self.count_f105()
        if f106:
            self.count_f106()
        if f107:
            self.count_f107()
        if f108:
            self.count_f108()
        if f109:
            self.count_f109()
        if f110:
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

    def count_f106(self, skip_end=False):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                n = len(words_tags_arr)
                for i in range(n):
                    if skip_end and i == n-1:
                        continue
                    pword = words_tags_arr[i].split('_')[0].lower()
                    ctag = words_tags_arr[i+1].split('_')[1] if i < n-1 else STOP
                    add_or_append(self.f106_count_dict, (pword, ctag))
                    
    def count_f107(self, skip_start=False):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                n = len(words_tags_arr)
                for i in range(n):
                    if skip_start and i == 0:
                        continue
                    nword = words_tags_arr[i].split('_')[0].lower()
                    ctag = words_tags_arr[i-1].split('_')[1] if i > 0 else BEGIN
                    add_or_append(self.f107_count_dict, (nword, ctag))

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


class Feature2Id:
    def __init__(self, file_path, feature_statistics, threshold, fix_threshold):
        self.file_path = file_path
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.fix_threshold = fix_threshold  # feature count threshold for prefix and suffix features
        self.total_features = 0  # Total number of features accumulated
        # Internal feature indexing
        self.f100_counter = 0
        self.f101_counter = 0
        self.f102_counter = 0
        self.f103_counter = 0
        self.f104_counter = 0
        self.f105_counter = 0
        self.f106_counter = 0
        self.f107_counter = 0
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
        self.f106_index_dict = OrderedDict()
        self.f107_index_dict = OrderedDict()
        self.f108_index_dict = OrderedDict()
        self.f109_index_dict = OrderedDict()
        self.f110_index_dict = OrderedDict()

    def initialize_index_dicts(self, f100, f101, f102, f103, f104, f105, f106,
                               f107, f108, f109, f110):
        """
        Initializes index dictionaries for features given in the list.
        :param f100: True if f100 should be initialized
        :param f101: True if f101 should be initialized
        :param f102: True if f102 should be initialized
        :param f103: True if f103 should be initialized
        :param f104: True if f104 should be initialized
        :param f105: True if f105 should be initialized
        :param f106: True if f105 should be initialized
        :param f107: True if f105 should be initialized
        :param f108: True if f108 should be initialized
        :param f109: True if f109 should be initialized
        :param f110: True if f110 should be initialized
        """
        if f100:
            self.initialize_f100_index_dict()
        if f101:
            self.initialize_f101_index_dict()
        if f102:
            self.initialize_f102_index_dict()
        if f103:
            self.initialize_f103_index_dict()
        if f104:
            self.initialize_f104_index_dict()
        if f105:
            self.initialize_f105_index_dict()
        if f106:
            self.initialize_f106_index_dict()
        if f107:
            self.initialize_f107_index_dict()
        if f108:
            self.initialize_f108_index_dict()
        if f109:
            self.initialize_f109_index_dict()
        if f110:
            self.initialize_f110_index_dict()

    def get_all_tags(self):
        """
        A quick way to access all tags the model "knows", i.e. passed threshold.
        :return: List of tags
        """
        return [tag for tag in self.f105_index_dict.keys()]

    def get_master_index(self):
        master_index = OrderedDict()
        master_index.update(self.f100_index_dict)
        master_index.update(self.f101_index_dict)
        master_index.update(self.f102_index_dict)
        master_index.update(self.f103_index_dict)
        master_index.update(self.f104_index_dict)
        master_index.update(self.f105_index_dict)
        master_index.update(self.f108_index_dict)
        master_index.update(self.f109_index_dict)
        master_index.update(self.f110_index_dict)
        return master_index

    def initialize_f100_index_dict(self):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = parse_lower(word_tag)
                    if (cword, ctag) not in self.f100_index_dict \
                            and self.feature_statistics.f100_count_dict[(cword, ctag)] >= self.threshold:
                        self.f100_index_dict[(cword, ctag)] = self.f100_counter + self.total_features
                        self.f100_counter += 1
        self.total_features += self.f100_counter

    # TODO weight this feature
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

    # TODO weight this feature
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
                                and self.feature_statistics.f102_count_dict[(suffix, ctag)] >= self.fix_threshold:
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
                            and self.feature_statistics.f103_count_dict[(pptag, ptag, ctag)] >= self.fix_threshold:
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
    
    def initialize_f106_index_dict(self, skip_end=False):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                n = len(words_tags_arr)
                for i in range(n):
                    if skip_end and i == n-1:
                        continue
                    pword = words_tags_arr[i].split('_')[0].lower()
                    ctag = words_tags_arr[i+1].split('_')[1] if i < n-1 else STOP
                    if (pword, ctag) not in self.f106_index_dict \
                            and self.feature_statistics.f106_count_dict[(pword, ctag)] >= self.threshold:
                        self.f106_index_dict[(pword, ctag)] = self.f106_counter + self.total_features
                        self.f106_counter += 1
        self.total_features += self.f106_counter
    
    def initialize_f107_index_dict(self, skip_start=False):
        with open(self.file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                n = len(words_tags_arr)
                for i in range(n):
                    if skip_start and i == 0:
                        continue
                    nword = words_tags_arr[i].split('_')[0].lower()
                    ctag = words_tags_arr[i-1].split('_')[1] if i > 0 else BEGIN
                    if (nword, ctag) not in self.f107_index_dict \
                            and self.feature_statistics.f107_count_dict[(nword, ctag)] >= self.threshold:
                        self.f107_index_dict[(nword, ctag)] = self.f107_counter + self.total_features
                        self.f107_counter += 1
        self.total_features += self.f107_counter
    
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

    def history_feature_representation(self, history, ctag):
        pword, cword, nword = history[4].lower(), history[0].lower(), history[3].lower()
        pptag, ptag = history[1], history[2]
        features = []
        has_upper = not history[0].islower()

        if (cword, ctag) in self.f100_index_dict:
            features.append(self.f100_index_dict[(cword, ctag)])

        for n in range(1, 5):
            if len(cword) <= n:
                break
            if (cword[:n], ctag) in self.f101_index_dict:
                features.append(self.f101_index_dict[(cword[:n], ctag)])
            if (cword[-n:], ctag) in self.f102_index_dict:
                features.append(self.f102_index_dict[(cword[-n:], ctag)])

        if (pptag, ptag, ctag) in self.f103_index_dict:
            features.append(self.f103_index_dict[(pptag, ptag, ctag)])

        if (ptag, ctag) in self.f104_index_dict:
            features.append(self.f104_index_dict[(ptag, ctag)])

        if ctag in self.f105_index_dict:
            features.append(self.f105_index_dict[ctag])

        if (pword, ctag) in self.f106_index_dict:
            features.append(self.f106_index_dict[(pword, ctag)])

        if (nword, ctag) in self.f107_index_dict:
            features.append(self.f107_index_dict[(nword, ctag)])

        if has_digit(cword) and (CONTAINS_DIGIT, ctag) in self.f108_index_dict:
            features.append(self.f108_index_dict[(CONTAINS_DIGIT, ctag)])

        if has_upper and (CONTAINS_UPPER, ctag) in self.f109_index_dict:
            features.append(self.f109_index_dict[(CONTAINS_UPPER, ctag)])

        if has_hyphen(cword) and (CONTAINS_HYPHEN, ctag) in self.f110_index_dict:
            features.append(self.f110_index_dict[(CONTAINS_HYPHEN, ctag)])

        return np.array(features)

