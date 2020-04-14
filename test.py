from collections import OrderedDict


BEGIN = '*B'
STOP = '*S'


class feature_statistics_class:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated
        self.f100_count_dict = OrderedDict()  # Init all features dictionaries
        self.f103_count_dict = OrderedDict()
        self.f104_count_dict = OrderedDict()
        self.f105_count_dict = OrderedDict()

    def count_f100(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line
                for word in words_arr:
                    # TODO think about normalizing capital letters (for example "the" and "The")
                    cur_word, cur_tag = word.split('_')
                    if (cur_word, cur_tag) not in self.f100_count_dict:
                        self.f100_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.f100_count_dict[(cur_word, cur_tag)] += 1

    def count_f103(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line

                # first word
                ctag = words_arr[0].split('_')[1]
                if (BEGIN, BEGIN, ctag) not in self.f103_count_dict:
                    self.f103_count_dict[(BEGIN, BEGIN, ctag)] = 1
                else:
                    self.f103_count_dict[(BEGIN, BEGIN, ctag)] += 1

                if len(words_arr) == 1:  # our job is done if the line only includes one word
                    return

                # second word
                ptag = ctag
                ctag = words_arr[1].split('_')[1]
                if (BEGIN, ptag, ctag) not in self.f103_count_dict:
                    self.f103_count_dict[(BEGIN, ptag, ctag)] = 1
                else:
                    self.f103_count_dict[(BEGIN, ptag, ctag)] += 1

                # third word and on
                for i in range(2, len(words_arr)):
                    pptag = ptag
                    ptag = ctag
                    ctag = words_arr[i].split('_')[1]
                    if (pptag, ptag, ctag) not in self.f103_count_dict:
                        self.f103_count_dict[(pptag, ptag, ctag)] = 1
                    else:
                        self.f103_count_dict[(pptag, ptag, ctag)] += 1

    def count_f104(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line

                # first word
                ctag = words_arr[0].split('_')[1]
                if (BEGIN, ctag) not in self.f104_count_dict:
                    self.f104_count_dict[(BEGIN, ctag)] = 1
                else:
                    self.f104_count_dict[(BEGIN, ctag)] += 1

                # second word and on
                for i in range(1, len(words_arr)):
                    ptag = ctag
                    ctag = words_arr[i].split('_')[1]
                    if (ptag, ctag) not in self.f104_count_dict:
                        self.f104_count_dict[(ptag, ctag)] = 1
                    else:
                        self.f104_count_dict[(ptag, ctag)] += 1

    def count_f105(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line
                for i in range(len(words_arr)):
                    ctag = words_arr[i].split('_')[1]
                    if ctag not in self.f105_count_dict:
                        self.f105_count_dict[ctag] = 1
                    else:
                        self.f105_count_dict[ctag] += 1


class feature2id_class:
    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.total_features = 0  # Total number of features accumulated
        # Internal feature indexing
        self.f100_counter = 0
        self.f103_counter = 0
        self.f104_counter = 0
        self.f105_counter = 0
        # Init all features dictionaries
        self.f100_index_dict = OrderedDict()
        self.f103_index_dict = OrderedDict()
        self.f104_index_dict = OrderedDict()
        self.f105_index_dict = OrderedDict()

    def initialize_f100_index_dict(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]
                for word in words_arr:
                    # TODO think about normalizing capital letters (for example "the" and "The")
                    cur_word, cur_tag = word.split('_')
                    if (cur_word, cur_tag) not in self.f100_index_dict \
                            and self.feature_statistics.f100_count_dict[(cur_word, cur_tag)] >= self.threshold:
                        self.f100_index_dict[(cur_word, cur_tag)] = self.f100_counter
                        self.f100_counter += 1
        self.total_features += self.f100_counter + 1  # TODO feature counter starts from one or zero?

    def initialize_f103_index_dict(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line

                # first word
                ctag = words_arr[0].split('_')[1]
                if (BEGIN, BEGIN, ctag) not in self.f103_index_dict \
                        and self.feature_statistics.f103_count_dict[(BEGIN, BEGIN, ctag)] >= self.threshold:
                    self.f103_index_dict[(BEGIN, BEGIN, ctag)] = self.f103_counter
                    self.f103_counter += 1

                if len(words_arr) == 1:
                    return

                # second word
                ptag = ctag
                ctag = words_arr[1].split('_')[1]
                if (BEGIN, ptag, ctag) not in self.f103_index_dict \
                        and self.feature_statistics.f103_count_dict[(BEGIN, ptag, ctag)] >= self.threshold:
                    self.f103_index_dict[(BEGIN, ptag, ctag)] = self.f103_counter
                    self.f103_counter += 1

                # third word and on
                for i in range(2, len(words_arr)):
                    pptag = ptag
                    ptag = ctag
                    ctag = words_arr[i].split('_')[1]
                    if (pptag, ptag, ctag) not in self.f103_index_dict \
                            and self.feature_statistics.f103_count_dict[(pptag, ptag, ctag)] >= self.threshold:
                        self.f103_index_dict[(pptag, ptag, ctag)] = self.f103_counter
                        self.f103_counter += 1

        self.total_features += self.f103_counter + 1  # TODO feature counter starts from one or zero?

    def initialize_f104_index_dict(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line

                # first word
                ctag = words_arr[0].split('_')[1]
                if (BEGIN, ctag) not in self.f104_index_dict \
                        and self.feature_statistics.f104_count_dict[(BEGIN, ctag)] >= self.threshold:
                    self.f104_index_dict[(BEGIN, ctag)] = self.f104_counter
                    self.f104_counter += 1

                if len(words_arr) == 1:
                    return

                # second word and on
                for i in range(1, len(words_arr)):
                    ptag = ctag
                    ctag = words_arr[i].split('_')[1]
                    if (ptag, ctag) not in self.f104_index_dict \
                            and self.feature_statistics.f104_count_dict[(ptag, ctag)] >= self.threshold:
                        self.f104_index_dict[(ptag, ctag)] = self.f104_counter
                        self.f104_counter += 1

        self.total_features += self.f104_counter + 1  # TODO feature counter starts from one or zero?

    def initialize_f105_index_dict(self, file_path):
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                if len(words_arr) == 0:
                    return
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line
                for i in range(len(words_arr)):
                    ctag = words_arr[i].split('_')[1]
                    if ctag not in self.f105_index_dict \
                            and self.feature_statistics.f105_count_dict[ctag] >= self.threshold:
                        self.f105_index_dict[ctag] = self.f105_counter
                        self.f105_counter += 1

        self.total_features += self.f105_counter + 1  # TODO feature counter starts from one or zero?


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


stats = feature_statistics_class()
stats.count_f100('train1.wtag')
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
print('')
