from preprocessing import *
from optimization import *
from inference import *
from metodot_ezer import *
from scipy.optimize import fmin_l_bfgs_b
import pickle
import numpy as np
from inference import memm_viterbi
from os import SEEK_END, remove

class Log_Linear_MEMM:
    def __init__(self):
        self.train_path = None
        self.feature_statistics = None
        self.feature2id = None
        self.weights = None
        self.dim = None
        self.threshold = None
        self.lam = None
        self.maxiter = None
        self.f100 = None
        self.f101 = None
        self.f102 = None
        self.f103 = None
        self.f104 = None
        self.f105 = None
        self.f106 = None
        self.f107 = None
        self.f108 = None
        self.f109 = None
        self.f110 = None

    # TODO check if this function is being used
    def get_all_tags(self):
        return self.feature2id.get_all_tags()

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

    def optimize(self, lam=0, maxiter=100, iprint=1, save_weights=True, weights_path='dumps/weights.pkl'):
        # initializing parameters for fmin_l_bfgs_b
        all_tags_list = self.feature2id.get_all_tags()
        all_histories, all_corresponding_tags = get_all_histories_ctags(self.train_path)  # abuse of notation :)
        features_list = calc_features_list(self.feature2id, all_histories, all_corresponding_tags)
        features_matrix = build_features_mat(self.feature2id, all_histories, all_tags_list)
        empirical_counts = calc_empirical_counts(features_list, self.dim)
        args = (self.dim, features_list, features_matrix, empirical_counts, lam)
        w_0 = np.random.random(self.dim)
        optimal_params = fmin_l_bfgs_b(func=calc_objective_and_grad, x0=w_0, args=args, maxiter=maxiter, iprint=iprint)

        if save_weights:
            with open(weights_path, 'wb') as f:
                pickle.dump(optimal_params, f)

        self.weights = optimal_params[0]

    def fit(self, train_path, threshold=10, lam=0, maxiter=100):
        """
        A simple interface to train a model.
        Only allows control on model hyper-parameters, no technical bullshit.
        :param train_path: A path for training data, *.wtag format.
        :param threshold: A threshold for the number of appearances of a parameter in corpus
        :param lam: lambda, regularization constant
        :param maxiter: the maximum amount of iterations of gradient descent
        """
        self.train_path = train_path
        self.preprocess()
        self.optimize(lam, maxiter)

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
