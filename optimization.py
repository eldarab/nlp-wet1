from metodot_ezer import *
from math import exp, log
from numpy.linalg import norm


def calc_objective_and_grad(v_i, feature_ids, histories_list, all_tags_list, reg_lambda,
                            empirical_counts, features_list):
    # TODO MATRITZA 2X2 onal_JJ
    # calculating linear term
    linear_term = 0
    for feature in features_list:
        linear_term += mult_sparse(v_i, feature)

    # calculating normalization_term
    normalization_term = 0
    expected_counts = 0
    for history in histories_list:
        tmp = 0
        for tag in all_tags_list:
            current_features = represent_history_with_features(feature_ids, history, tag)
            tmp += exp(mult_sparse(v_i, current_features))
        normalization_term += log(tmp)  # uses natural logarithm


    # calculating regularization
    regularization = 0.5 * reg_lambda * (norm(v_i) ** 2)  # l2 norm

    # calculating expected_counts

    for history in histories_list:
        tmp = 0
        for tag in all_tags_list:


    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad

