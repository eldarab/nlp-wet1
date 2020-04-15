from preprocessing import feature_statistics_class, feature2id_class
from metodot_ezer import get_all_histories_ctags, calc_features_list, build_features_mat, calc_empirical_counts
from optimization import calc_objective_and_grad
from scipy.optimize import fmin_l_bfgs_b
import pickle
import numpy as np
from inference import memm_viterbi


TRAIN_PATH = 'data/debugging_dataset.wtag'


def learn(file_path):
    # preprocessing
    statistics = feature_statistics_class(file_path)
    feature2id = feature2id_class(file_path, statistics, threshold=10)

    # initializing parameters for fmin_l_bfgs_b
    dim = feature2id.total_features
    all_tags_list = feature2id.get_all_tags()
    all_histories, all_corresponding_tags = get_all_histories_ctags(file_path)  # abuse of notation :)
    features_list = calc_features_list(feature2id, all_histories, all_corresponding_tags)
    features_matrix = build_features_mat(feature2id, all_histories, all_tags_list)
    reg_lambda = 0
    empirical_counts = calc_empirical_counts(features_list, dim)

    args = (dim, features_list, features_matrix, reg_lambda, empirical_counts)
    w_0 = np.random.random(dim)
    optimal_params = fmin_l_bfgs_b(func=calc_objective_and_grad, x0=w_0, args=args, maxiter=10, iprint=99)
    weights = optimal_params[0]

    # running optimization
    weights_path = 'pickelim/trained_weights_data_i.pkl'  # i identifies which dataset this is trained on
    with open(weights_path, 'wb') as f:
        pickle.dump(optimal_params, f)

    #### In order to load pre-trained weights, just use the next code: ####
    #                                                                     #
    # with open(weights_path, 'rb') as f:                                 #
    #   optimal_params = pickle.load(f)                                   #
    # pre_trained_weights = optimal_params[0]                             #
    #                                                                     #
    #######################################################################


def predict(file_path):
    statistics = feature_statistics_class(TRAIN_PATH)
    feature2id = feature2id_class(TRAIN_PATH, statistics, threshold=10)

    with open("pickelim/trained_weights_data_i.pkl", 'rb') as f:
        optimal_params = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    test = "Hadar went to the mall and bought some eggs ."
    return memm_viterbi(feature2id, pre_trained_weights, feature2id.get_all_tags(), test)


if __name__ == '__main__':
    tags = predict(TRAIN_PATH)
    pass

