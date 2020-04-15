import pickle

from metodot_ezer import *
from preprocessing import *
from math import exp, log
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b


TRAIN_PATH = 'data/train1.wtag'


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
