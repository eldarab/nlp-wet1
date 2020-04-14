from collections import OrderedDict

BEGIN = '*B'
STOP = '*S'


class feature_statistics_class:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated
        self.f100_count_dict = OrderedDict()  # Init all features dictionaries
        self.f101_count_dict = OrderedDict()
        self.f102_count_dict = OrderedDict()
        self.f103_count_dict = OrderedDict()
        self.f104_count_dict = OrderedDict()
        self.f105_count_dict = OrderedDict()
        self.f109_count_dict = OrderedDict()

    def count_f100(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    cword, ctag = parse_lower(word)
                    add_or_append(self.f100_count_dict, (cword, ctag))

    def count_f101(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    cword, ctag = parse_lower(word)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        add_or_append(self.f101_count_dict, (cword[:n], ctag))

    def count_f102(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    cword, ctag = parse_lower(word)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        add_or_append(self.f102_count_dict, (cword[-n:], ctag))

    def count_f103(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                ptag = BEGIN
                ctag = BEGIN
                for word in words_arr:
                    pptag = ptag
                    ptag = ctag
                    ctag = word.split('_')[1]
                    add_or_append(self.f103_count_dict, (pptag, ptag, ctag))

    def count_f104(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                ctag = BEGIN
                for word in words_arr:
                    ptag = ctag
                    ctag = word.split('_')[1]
                    add_or_append(self.f104_count_dict, (ptag, ctag))

    def count_f105(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    ctag = word.split('_')[1]
                    add_or_append(self.f105_count_dict, ctag)

    def count_f109(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_tags_arr = get_words_arr(line)
                for word_tag in words_tags_arr:
                    cword, ctag = word_tag.split('_')
                    if cword[0].isupper() and cword[1:].islower():
                        add_or_append(self.f109_count_dict, (cword, ctag))


class feature2id_class:
    def __init__(self, feature_statistics, threshold):
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
        # Init all features dictionaries
        self.f100_index_dict = OrderedDict()
        self.f101_index_dict = OrderedDict()
        self.f102_index_dict = OrderedDict()
        self.f103_index_dict = OrderedDict()
        self.f104_index_dict = OrderedDict()
        self.f105_index_dict = OrderedDict()

    def initialize_f100_index_dict(self, file_path):
        self.f100_counter = self.total_features
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    cword, ctag = parse_lower(word)
                    if (cword, ctag) not in self.f100_index_dict \
                            and self.feature_statistics.f100_count_dict[(cword, ctag)] >= self.threshold:
                        self.f100_index_dict[(cword, ctag)] = self.f100_counter
                        self.f100_counter += 1
        self.total_features += self.f100_counter

    def initialize_f101_index_dict(self, file_path):
        self.f101_counter = self.total_features
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    cword, ctag = parse_lower(word)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        prefix = cword[:n]
                        if (prefix, ctag) not in self.f101_index_dict \
                                and self.feature_statistics.f101_count_dict[(prefix, ctag)] >= self.threshold:
                            self.f101_index_dict[(prefix, ctag)] = self.f101_counter
                            self.f101_counter += 1
        self.total_features += self.f101_counter

    def initialize_f102_index_dict(self, file_path):
        self.f102_counter = self.total_features
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    cword, ctag = parse_lower(word)
                    for n in range(1, 5):
                        if len(cword) <= n:
                            break
                        suffix = cword[-n:]
                        if (suffix, ctag) not in self.f102_index_dict \
                                and self.feature_statistics.f102_count_dict[(suffix, ctag)] >= self.threshold:
                            self.f101_index_dict[(suffix, ctag)] = self.f102_counter
                            self.f102_counter += 1
        self.total_features += self.f102_counter

    def initialize_f103_index_dict(self, file_path):
        self.f103_counter = self.total_features
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                ptag = BEGIN
                ctag = BEGIN
                for word in words_arr:
                    pptag = ptag
                    ptag = ctag
                    ctag = word.split('_')[1]
                    if (pptag, ptag, ctag) not in self.f103_index_dict \
                            and self.feature_statistics.f103_count_dict[(pptag, ptag, ctag)] >= self.threshold:
                        self.f103_index_dict[(pptag, ptag, ctag)] = self.f103_counter
                        self.f103_counter += 1
        self.total_features += self.f103_counter

    def initialize_f104_index_dict(self, file_path):
        self.f104_counter = self.total_features
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                ctag = BEGIN
                for word in words_arr:
                    ptag = ctag
                    ctag = word.split('_')[1]
                    if (ptag, ctag) not in self.f104_index_dict \
                            and self.feature_statistics.f104_count_dict[(ptag, ctag)] >= self.threshold:
                        self.f104_index_dict[(ptag, ctag)] = self.f104_counter
                        self.f104_counter += 1
        self.total_features += self.f104_counter

    def initialize_f105_index_dict(self, file_path):
        self.f105_counter = self.total_features
        with open(file_path) as f:
            for line in f:
                words_arr = get_words_arr(line)
                for word in words_arr:
                    ctag = word.split('_')[1]
                    if ctag not in self.f105_index_dict \
                            and self.feature_statistics.f105_count_dict[ctag] >= self.threshold:
                        self.f105_index_dict[ctag] = self.f105_counter
                        self.f105_counter += 1
        self.total_features += self.f105_counter


def represent_history_with_features(history, f100_index_dict, f103_index_dict,
                                    f104_index_dict, f105_index_dict):
    pword, cword, nword = history[5], history[0], history[4]
    pptag, ptag, ctag = history[1], history[2], history[3]
    features = []

    if (cword, ctag) in f100_index_dict:
        features.append(f100_index_dict[(cword, ctag)])

    if ctag in f105_index_dict:
        features.append(f105_index_dict[ctag])

    if (ptag, ctag) in f104_index_dict:
        features.append(f104_index_dict[(ptag, ctag)])

    if (pptag, ptag, ctag) in f103_index_dict:
        features.append(f103_index_dict[(pptag, ptag, ctag)])

    return features


def add_or_append(dict, item):
    if item not in dict:
        dict[item] = 1
    else:
        dict[item] += 1


def parse_lower(word_tag):
    word, tag = word_tag.split('_')
    return word.lower(), tag


def get_words_arr(line):
    words_arr = line.split(' ')
    if len(words_arr) == 0:
        return
    words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line
    return words_arr


if __name__ == '__main__':
    stats = feature_statistics_class()
    stats.count_f100('train1.wtag')
    stats.count_f101('train1.wtag')
    stats.count_f102('train1.wtag')
    stats.count_f103('train1.wtag')
    stats.count_f104('train1.wtag')
    stats.count_f105('train1.wtag')
    ids = feature2id_class(stats, 4)
    ids.initialize_f100_index_dict('train1.wtag')
    ids.initialize_f103_index_dict('train1.wtag')
    ids.initialize_f104_index_dict('train1.wtag')
    ids.initialize_f105_index_dict('train1.wtag')
    history1 = ('went', '*B', 'NN', 'VBD', 'Eldar', 'to')
    rep = represent_history_with_features(history1, ids.f100_index_dict, ids.f103_index_dict,
                                          ids.f104_index_dict, ids.f105_index_dict)
    pass
