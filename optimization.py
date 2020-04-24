from preprocessing import *
from math import exp, log
from numpy.linalg import norm


# TODO explain parameters
def calc_objective_and_grad(v_i, dim, features_list, features_matrix, empirical_counts, reg_lambda):
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
    # removed feature_ids and using dim instead

    #   Objective Function

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
    temp = norm(v_i)
    regularization = 0.5 * reg_lambda * (norm(v_i) ** 2)  # l2 norm

    #   Gradient Function

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
