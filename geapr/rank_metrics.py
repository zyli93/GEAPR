"""
The code for `apk` and `mapk` is from https://github.com/benhamner/Metrics
Many thanks to Ben Hamner!

"""
import numpy as np


def gen_bin_indicator(gt, n_item):
    indicators = []
    for i in range(len(gt)):
        tmp = np.zeros(n_item)
        np.put(tmp, gt[i], 1)
        indicators.append(tmp)
    return np.stack(indicators, axis=0)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)


    try:
        if not actual:
            return 0.0
    except:
        print(actual)
        print(type(actual))
        raise ValueError()

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def precision_at_k(actual, predicted, k):
    """precision at k
    Args:
        actual - list of list
        predicted - list of list
    Return:
        prec@k
    """
    assert len(actual) == len(predicted), "prec@k inconsistent length"
    assert k < len(predicted[0]), "TOO Big K"
    prec_at_k_list = [len(set(actual[i]) & set(predicted[i][:k])) / k
                      for i in range(len(actual))]
    return sum(prec_at_k_list) / len(prec_at_k_list)


def recall_at_k(actual, predicted, k):
    """recall at k
    Args:
        actual - list of list
        predicted - list of list
    Return:
        recall@k
    """
    assert len(actual) == len(predicted), "prec@k inconsistent length"
    assert k < len(predicted), "TOO Big K"
    recall_at_k_list = [len(set(actual[i]) & set(predicted[i][:k])) / len(actual[i])
                        for i in range(len(actual))]
    return sum(recall_at_k_list) / len(recall_at_k_list)


def metrics_poi(gt, pred_scores, k_list):
    """a bundle of four metrics: prec@k, recall@k, map@k, and ndcg@k

    Args:
        gt - list of lists of the ground-truth
        pred_scores - list of list of the predicted scores
        k_list - a list of k

    Returns:
        eval_dict - dicionary of evaluation: 
            {k:
                {"prec":x, "recall":x, "map":x, "ndcg":x}
             ...}
    """
    eval_dict = dict()
    pred_scores[:, 0] = 1e9
    print("\t[Evaluation] Running argsort ...")
    pred_ranking = np.flip(np.argsort(pred_scores, axis=1), axis=1).tolist()

    for k in k_list:
        # print("\t[Evaluation] {}".format(k))
        eval_dict[k] = {
            "prec_ak": precision_at_k(actual=gt, predicted=pred_ranking, k=k)
            , "recall_ak": recall_at_k(actual=gt, predicted=pred_ranking, k=k)
            , "mapk": mapk(actual=gt, predicted=pred_ranking, k=k)
            }
    return eval_dict


def build_metrics_msgs(eval_dict):
    """build msg, not used"""
    return ["k-{}".format(k) + 
                " ".join(["{}-{:.6f}".format(k, v) for k, v in eval_dict[k]])
            for k in eval_dict.keys()]



