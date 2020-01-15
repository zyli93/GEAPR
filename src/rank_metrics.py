"""
The code below is from https://gist.github.com/bwhite/3726239
              and from https://github.com/benhamner/Metrics
Many thanks to Brandyn White and Ben Hamner!

`dcg_at_k` and `ndcg_at_k` are from the former source.
`apk` and `mapk` are from the later source.

Many thanks again!

"""
import numpy as np


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


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

    if not actual:
        return 0.0

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
    assert k < len(predicted), "TOO Big K"
    prec_at_k_list = [len(set(actual[i]) & set(predicted[i][:k])) / k
                      for i in range(len(actual))]
    return sum(prec_at_k_list) / len(prec_at_k_list)


def recall_at_K(actual, predicted, k):
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


def metrics_poi_rec():
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
    pred_ranking = np.argsort(pred_scores, axis=1)
    ndcg_ranking = np.isin(pred_scores, gt).astype(np.int32)

    for k in k_list:
        # mean average precision, mask out the empty id slot
        eval_dict[k] = {
            "mapk": mapk(actual=gt, predicted=pred_ranking, k=k),
            "prec_ak": precision_at_k(actual=gt, predicted=pred_ranking, k=k),
            "recall_ak": recall_at_k(actual=gt, predicted=pred_ranking, k=k),
            "ndcgk": ndcg_at_k(r=pred_ranking, k=k)
            }
    return eval_dict









