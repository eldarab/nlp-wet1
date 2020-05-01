from preprocessing import *
from optimization import *
from inference import *
from metodot_ezer import *
from scipy.optimize import fmin_l_bfgs_b
import pickle
import numpy as np
from inference import memm_viterbi
from time import strftime
from os import SEEK_END, remove


class Log_Linear_MEMM:
    def __init__(self, threshold=10, lam=0, maxiter=100, f100=True, f101=True, f102=True, f103=True, f104=True,
                 f105=True, f106=True, f107=True, f108=True, f109=True, f110=True):
        self.train_path = None
        self.feature_statistics = None
        self.feature2id = None
        self.weights = None
        self.lbfgs_result = None
        self.dim = None
        self.threshold = threshold
        self.lam = lam
        self.maxiter = maxiter
        self.f100 = f100
        self.f101 = f101
        self.f102 = f102
        self.f103 = f103
        self.f104 = f104
        self.f105 = f105
        self.f106 = f106
        self.f107 = f107
        self.f108 = f108
        self.f109 = f109
        self.f110 = f110

    # TODO check if this function is being used
    def get_all_tags(self):
        return self.feature2id.get_all_tags()

    # TODO load model instead of weights
    def load_weights(self, weights_path='dumps/weights.pkl'):
        with open(weights_path, 'rb') as f:
            optimal_params = pickle.load(f)
        self.weights = optimal_params[0]

    def preprocess(self):
        self.feature_statistics = FeatureStatisticsClass(self.train_path)
        self.feature_statistics.count_features(self.f100, self.f101, self.f102, self.f103, self.f104, self.f105,
                                               self.f106, self.f107, self.f108, self.f109, self.f110)
        self.feature2id = Feature2Id(self.train_path, self.feature_statistics, self.threshold)
        self.feature2id.initialize_index_dicts(self.f100, self.f101, self.f102, self.f103, self.f104, self.f105,
                                               self.f106, self.f107, self.f108, self.f109, self.f110)
        self.dim = self.feature2id.total_features

    def optimize(self, iprint=1):
        # initializing parameters for fmin_l_bfgs_b
        all_tags_list = self.feature2id.get_all_tags()
        all_histories, all_corresponding_tags = get_all_histories_ctags(self.train_path)  # abuse of notation :)
        features_list = calc_features_list(self.feature2id, all_histories, all_corresponding_tags)
        features_matrix = build_features_mat(self.feature2id, all_histories, all_tags_list)
        empirical_counts = calc_empirical_counts(features_list, self.dim)
        args = (self.dim, features_list, features_matrix, empirical_counts, self.lam)
        w_0 = np.random.random(self.dim)
        optimal_params = fmin_l_bfgs_b(func=calc_objective_and_grad, x0=w_0, args=args, maxiter=self.maxiter,
                                       iprint=iprint)
        self.lbfgs_result = optimal_params
        self.weights = optimal_params[0]

    def fit(self, train_path):
        """
        A simple interface to train a model.
        :param train_path: A path for training data, *.wtag format.
        """
        self.train_path = train_path
        self.preprocess()
        self.optimize()

    def save(self, filename='model_' + strftime("%Y-%m-%d_%H-%M-%S")):
        pkl_path = 'dumps/' + filename + '.pkl'
        txt_path = 'dumps/' + filename + '.txt'
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)
        with open(txt_path, 'w') as f:
            f.write(filename + '\n')
            f.write('train_path = ' + str(self.train_path) + '\n')
            f.write('dim = ' + str(self.dim) + '\n')
            f.write('threshold = ' + str(self.threshold) + '\n')
            f.write('lam = ' + str(self.lam) + '\n')
            f.write('maxiter = ' + str(self.maxiter) + '\n')
            f.write('f100 = ' + str(self.f100) + '\n')
            f.write('f101 = ' + str(self.f101) + '\n')
            f.write('f102 = ' + str(self.f102) + '\n')
            f.write('f103 = ' + str(self.f103) + '\n')
            f.write('f104 = ' + str(self.f104) + '\n')
            f.write('f105 = ' + str(self.f105) + '\n')
            f.write('f106 = ' + str(self.f106) + '\n')
            f.write('f107 = ' + str(self.f107) + '\n')
            f.write('f108 = ' + str(self.f108) + '\n')
            f.write('f109 = ' + str(self.f109) + '\n')
            f.write('f110 = ' + str(self.f110) + '\n')

    def predict(self, input_data, beam_size=5):
        """
        Generates a prediction for a given input. Input can be either a sentence (string) or a file path.
        File can be in either .wtag or .words format.
        :param beam_size: a parameter of the viterbi
        :param input_data: string or file path
        """
        # TODO maybe save prediction file as parameter of the model?
        if len(input_data) > 6 and input_data[-6:] == '.words':
            return self.predict_file(input_data, beam_size)

        if len(input_data) > 5 and input_data[-5:] == '.wtag':
            temp_file = r'data\temp.words'
            clean_tags(input_data, temp_file)
            predictions = self.predict_file(temp_file, beam_size)
            remove(temp_file)
            return predictions

        else:
            return memm_viterbi(self.feature2id, self.weights, input_data, beam_size)

    def predict_file(self, file, beam_size):
        with open(file, 'r') as in_file:
            predictions = []
            for line in in_file:
                line_predictions = []
                words = line.split()
                prediction = memm_viterbi(self.feature2id, self.weights, line, beam_size)
                for word, pred in zip(words, prediction):
                    line_predictions.append(word + '_' + pred)
                predictions.append(line_predictions)
        return predictions
