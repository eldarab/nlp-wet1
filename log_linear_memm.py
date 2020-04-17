from preprocessing import *
from optimization import *
from inference import *
from metodot_ezer import *
from scipy.optimize import fmin_l_bfgs_b
import pickle
import numpy as np
from inference import memm_viterbi


class Log_Linear_MEMM:
    def __init__(self):
        self.train_path = None
        self.feature_statistics = None
        self.feature2id = None
        self.weights = None
        self.dim = None

    def load_weights(self, weights_path='dumps/weights.pkl'):
        with open(weights_path, 'rb') as f:
            optimal_params = pickle.load(f)
        self.weights = optimal_params[0]

    def preprocess(self, threshold=10):
        self.feature_statistics = feature_statistics_class(self.train_path)
        self.feature2id = feature2id_class(self.train_path, self.feature_statistics, threshold)
        self.feature2id.initialize_index_dicts()
        self.dim = self.feature2id.total_features

    def optimize(self, lam=0, maxiter=1000, iprint=1, save_weights=True, weights_path='dumps/weights.pkl'):
        # initializing parameters for fmin_l_bfgs_b
        all_tags_list = self.feature2id.get_all_tags()
        all_histories, all_corresponding_tags = get_all_histories_ctags(self.train_path)  # abuse of notation :)
        features_list = calc_features_list(self.feature2id, all_histories, all_corresponding_tags)
        features_matrix = build_features_mat(self.feature2id, all_histories, all_tags_list)
        empirical_counts = calc_empirical_counts(features_list, self.dim)

        args = (self.dim, features_list, features_matrix, lam, empirical_counts)
        w_0 = np.random.random(self.dim)
        optimal_params = fmin_l_bfgs_b(func=calc_objective_and_grad, x0=w_0, args=args, maxiter=maxiter, iprint=iprint)

        if save_weights:
            with open(weights_path, 'wb') as f:
                pickle.dump(optimal_params, f)

        self.weights = optimal_params[0]

    def fit(self, train_path, threshold=10, lam=0):
        """
        A simple interface to train a model.
        Only allows control on model hyper-parameters, no technical bullshit.
        :param train_path: A path for training data, *.wtag format.
        :param threshold: A threshold for the number of appearances of a parameter in corpus
        :param lam: lambda, regularization constant
        """
        self.train_path = train_path
        self.preprocess(threshold)
        self.optimize(lam)

    def predict(self, input_data, predictions_path='predictions/predictions.wtag'):
        """
        Generates a prediction for a given input. Input can be either a sentence (string) or a file path.
        File has to be in *.wtag format.
        :param predictions_path: Saves predictions to path if input was a file
        :param input_data: string or file path
        """
        if len(input_data) > 5 and input_data[-5:] == '.wtag':
            pass
            # TODO make predictions on entire file and save predictions to predictions_path
        else:
            return memm_viterbi(self.feature2id, self.weights, self.feature2id.get_all_tags(), input_data)
