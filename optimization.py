from preprocessing import *
from math import exp, log
from numpy.linalg import norm

from concurrent.futures import ThreadPoolExecutor


def calc_objective_and_grad_old(v_i, dim, features_list, features_matrix, empirical_counts, reg_lambda):
    """
    Generates objective and gradient to use in fmin_l_bfgs_b in a single iteration
    :param v_i: [[DENSE]] Parameter to optimize at iteration i
    :param dim: [[SCALAR]] the dimension of the space we optimize in
    :param features_list: [[SPARSE]] A list of the sparse feature representation of all histories in corpus, i.e. f(xi,yi)
    :param features_matrix: [[SPARSE]] A matrix containing sparse feature representation of all histories combined with
    all tags in corpus, i.e. f(xi,y') for each y' in tags
    :param empirical_counts: [[DENSE]] A dense representation of empirical_counts
    :param reg_lambda: [[SCALAR]] Hyper-parameter that controls regularization
    :return: A tuple of the likelihood (objective) and it's gradient to pass to fmin_l_bfgs_b
    """
    # TODO replace mult_sparse with np arrays or sparse arrays and matrix multiplications

    #       Objective Function

    # calculating linear term
    linear_term = 0
    for feature in features_list:
        linear_term += mult_sparse(v_i, feature)

    # feat_mat is a np array of all of the features from the train data
    # linear_term = np.sum(v_i @ feat_mat)

    # calculating normalization_term
    # TODO consider implementing using matrix-vector multiplication instead of mult_sparse
    # features matrix can be either a 3 dimensional array or a list of 2 dimensional arrays
    normalization_term = 0
    for history in features_matrix:
        tmp = 0
        for feature in history:
            tmp += exp(mult_sparse(v_i, feature))
        normalization_term += log(tmp)  # natural logarithm

    # calculating regularization
    regularization = 0.5 * reg_lambda * (norm(v_i) ** 2)  # l2 norm

    #       Gradient Function

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


def calc_linear_term(v_i, features_list):
    linear_term = 0
    for feature in features_list:
        linear_term += mult_sparse(v_i, feature)
    return linear_term


def calc_normalization_term(v_i, features_matrix):
    normalization_term = 0
    for history in features_matrix:
        tmp = 0
        for feature in history:
            tmp += exp(mult_sparse(v_i, feature))
        normalization_term += log(tmp)  # natural logarithm
    return normalization_term


def calc_regularization(v_i, reg_lambda):
    return 0.5 * reg_lambda * (norm(v_i) ** 2)


def calc_expected_counts(v_i, dim, features_matrix):
    expected_counts = 0
    for history in features_matrix:
        numerator = np.zeros(dim)
        denominator = 0
        for feature in history:
            denominator += exp(mult_sparse(v_i, feature))
            numerator += exp(mult_sparse(v_i, feature)) * sparse_to_dense(feature, dim)
        expected_counts += numerator / denominator
    return expected_counts


def calc_objective_and_grad(v_i, dim, features_list, features_matrix, empirical_counts,
                            reg_lambda):
    """
    Generates objective and gradient to use in fmin_l_bfgs_b in a single iteration
    :param v_i: [[DENSE]] Parameter to optimize at iteration i
    :param dim: [[SCALAR]] the dimension of the space we optimize in
    :param features_list: [[SPARSE]] A list of the sparse feature representation of all histories in corpus, i.e. f(xi,yi)
    :param features_matrix: [[SPARSE]] A matrix containing sparse feature representation of all histories combined with
    all tags in corpus, i.e. f(xi,y') for each y' in tags
    :param empirical_counts: [[DENSE]] A dense representation of empirical_counts
    :param expected_counts_vec: [[DENSE]] A dense representation of expected_counts numerator, without p(y'|x;v) scalar
    :param reg_lambda: [[SCALAR]] Hyper-parameter that controls regularization
    :return: A tuple of the likelihood (objective) and it's gradient to pass to fmin_l_bfgs_b
    """

    #       Objective Function
    # calculating linear term
    linear_term = calc_linear_term(v_i, features_list)

    # feat_mat is a np array of all of the features from the train data
    # linear_term = np.sum(v_i @ feat_mat)

    normalization_term = calc_normalization_term(v_i, features_matrix)
    regularization = calc_regularization(v_i, reg_lambda)  # l2 norm

    #       Gradient Function
    expected_counts = calc_expected_counts(v_i, dim, features_matrix)
    regularization_grad = reg_lambda * v_i

    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad


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
