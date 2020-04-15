import pickle

from metodot_ezer import *
from preprocessing import *
from math import exp, log
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b


TRAIN_PATH = 'debugging_dataset.wtag'


# TODO explain parameters
def calc_objective_and_grad(v_i, dim, features_list, features_matrix, empirical_counts, reg_lambda):
    # removed feature_ids and using dim instead

    # calculating linear term
    linear_term = 0
    for feature in features_list:
        linear_term += mult_sparse(v_i, feature)

    # calculating normalization_term
    # TODO consider implementing using matrix-vector multiplication instead of mult_sprase
    normalization_term = 0
    for history in features_matrix:
        tmp = 0
        for feature in history:
            tmp += exp(mult_sparse(v_i, feature))
        normalization_term += log(tmp)  # natural logarithm

    # calculating regularization
    regularization = 0.5 * reg_lambda * (norm(v_i) ** 2)  # l2 norm

    # calculating expected_counts
    expected_counts = 0
    for history in features_matrix:
        numerator = np.zeros(dim)
        denominator = 0
        for feature in history:
            denominator += exp(mult_sparse(v_i, feature))
            numerator += exp(mult_sparse(v_i, feature)) * sparse_to_dense(feature, dim)
        expected_counts += numerator / denominator

    # calculating regularization_grad
    regularization_grad = reg_lambda * v_i

    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad


if __name__ == '__main__':
    # preprocessing
    statistics = feature_statistics_class(TRAIN_PATH)
    feature2id = feature2id_class(TRAIN_PATH, statistics, threshold=10)

    # initializing parameters for fmin_l_bfgs_b
    dim = feature2id.total_features
    all_tags_list = feature2id.get_all_tags()
    all_histories, all_ctags = get_all_histories_ctags(TRAIN_PATH)  # abuse of notation :)
    features_list = calc_features_list(feature2id, all_histories, all_ctags)
    features_matrix = build_features_mat(feature2id, all_histories, all_tags_list)
    reg_lambda = 0
    empirical_counts = calc_empirical_counts(features_list, dim)

    args = (dim, features_list, features_matrix, reg_lambda, empirical_counts)
    w_0 = np.random.random(dim)
    optimal_params = fmin_l_bfgs_b(func=calc_objective_and_grad, x0=w_0, args=args, maxiter=10, iprint=50)
    weights = optimal_params[0]

    # running optimization
    weights_path = 'pickelim/trained_weights_data_i.pkl'  # i identifies which dataset this is trained on
    with open(weights_path, 'wb') as f:
        pickle.dump(optimal_params, f)

    print(weights)
    #### In order to load pre-trained weights, just use the next code: ####
    #                                                                     #
    # with open(weights_path, 'rb') as f:                                 #
    #   optimal_params = pickle.load(f)                                   #
    # pre_trained_weights = optimal_params[0]                             #
    #                                                                     #
    #######################################################################
