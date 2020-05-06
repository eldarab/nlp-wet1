from math import exp, log

import numpy as np
from numpy.linalg import norm

from auxiliary_functions import multiply_sparse, exp_multiply_sparse, sparse_to_dense


def calc_empirical_counts(features_list, dim):
    empirical_counts = np.zeros(dim)
    for feature in features_list:
        empirical_counts += sparse_to_dense(feature, dim)
    return empirical_counts


def calc_linear_term(v_i, features_list):
    linear_term = 0
    for feature in features_list:
        linear_term += multiply_sparse(v_i, feature)
    return linear_term


def calc_normalization_term_old(v_i, features_matrix):
    normalization_term = 0
    for history in features_matrix:
        tmp = 0
        for feature in history:
            tmp += exp(multiply_sparse(v_i, feature))
        normalization_term += log(tmp)  # natural logarithm
    return normalization_term


def calc_normalization_term_new(v_i, features_matrix):
    normalization_term = 0
    exp_v_i = np.exp(v_i)
    for history in features_matrix:
        tmp = 0
        for feature in history:
            tmp += exp_multiply_sparse(exp_v_i, feature)
        normalization_term += log(tmp)  # natural logarithm
    return normalization_term


def calc_regularization(v_i, reg_lambda):
    return 0.5 * reg_lambda * (norm(v_i) ** 2)


def calc_expected_counts_old(v_i, dim, features_matrix):
    expected_counts = np.zeros(dim)
    for i in range(len(features_matrix)):
        denominator = 0
        numerator = np.zeros(dim)
        for feature in features_matrix[i]:
            temp = exp(multiply_sparse(v_i, feature))
            denominator += temp
            numerator += temp * sparse_to_dense(feature, dim)

        expected_counts += numerator / denominator
    return expected_counts


def calc_expected_counts_new(v_i, dim, features_matrix):
    expected_counts = np.zeros(dim)
    exp_v_i = np.exp(v_i)
    for history in features_matrix:
        denominator = 0
        # index_weights = {}
        numerator = np.zeros(dim)
        for feature in history:
            temp = exp_multiply_sparse(exp_v_i, feature)
            denominator += temp
            for f in feature:
                numerator[f] += temp
        expected_counts += numerator / denominator
    return expected_counts


def calc_objective(v_i, dim, features_list, features_matrix, empirical_counts, reg_lambda, use_new):
    linear_term = calc_linear_term(v_i, features_list)
    # normalization_term = calc_normalization_term_new(v_i, features_matrix) if use_new \
    #     else calc_normalization_term_old(v_i, features_matrix)
    normalization_term = calc_normalization_term_new(v_i, features_matrix)

    regularization = calc_regularization(v_i, reg_lambda)

    likelihood = linear_term - normalization_term - regularization
    return -1 * likelihood


def calc_gradient(v_i, dim, features_list, features_matrix, empirical_counts, reg_lambda, use_new):
    # expected_counts = calc_expected_counts_new(v_i, dim, features_matrix) if use_new \
    #     else calc_expected_counts_old(v_i, dim, features_matrix)
    expected_counts = calc_expected_counts_new(v_i, dim, features_matrix)

    regularization_grad = reg_lambda * v_i

    gradient = empirical_counts - expected_counts - regularization_grad
    return (-1) * gradient


# TODO compare this to regular version and consider multiprocessing
"""
def calc_objective_and_grad_with_threading(v_i, dim, features_list, features_matrix, empirical_counts,
                                           reg_lambda):
    with ThreadPoolExecutor(max_workers=4) as executor:
        #       Objective Function
        # calculating linear term
        linear_term = executor.submit(calc_linear_term, v_i, features_list).result()

        normalization_term = executor.submit(calc_normalization_term, v_i, features_matrix).result()
        regularization = executor.submit(calc_regularization, v_i, reg_lambda).result()

        #       Gradient Function
        expected_counts = executor.submit(calc_expected_counts, v_i, dim, features_matrix).result()
        regularization_grad = reg_lambda * v_i

        likelihood = linear_term - normalization_term - regularization
        grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad
"""
