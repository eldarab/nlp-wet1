from collections import OrderedDict
import string


class feature_statistics_class:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        # ---Add more count dictionaries here---

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                words_arr = line.split(' ')
                words_arr[-1] = words_arr[-1][:-1]  # removing \n from end of line
                for word in words_arr:
                    cur_word, cur_tag = word.split('_')
                    if (cur_word, cur_tag) not in self.words_tags_dict:  # TODO why that this would be here from the beginning?
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1


class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = collections.OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = split(splited_words[word_idx], '_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.feature_statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #


test_class = feature_statistics_class()
test_class.get_word_tag_pair_count('train1.wtag')
