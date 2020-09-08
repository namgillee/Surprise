"""
The :mod:`surprise.accuracy` module provides tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems


def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mse(predictions, verbose=True):
    """Compute MSE (Mean Squared Error).

    .. math::
        \\text{MSE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse_ = np.mean([float((true_r - est)**2)
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MSE: {0:1.4f}'.format(mse_))

    return mse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=True):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp


def precision(predictions, binary_threshold=4, greater_than_threshold=True, zero_division=1, verbose=True):
    """Compute precision for thresholded predictions.

    .. math::
        \\text{precision} = \\frac{tp}{tp + fp}

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        binary_threshold
            threshold for binary classification
        greater_than_threshold
            If True, an estimated rating is considered positive iff est > binary_threshold,
            and a true rating is considered positive iff true_r > binary_threshold.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Precision of thresholded predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    tp_ = np.sum([(true_r > positive_threshold and est > positive_threshold)
                    for (_, _, true_r, est, _) in predictions])

    fp_ = np.sum([(true_r <= positive_threshold and est > positive_threshold)
                    for (_, _, true_r, est, _) in predictions])

    prec_ = tp_/(tp_ + fp_) if (tp_ + fp_) > 0 else zero_division

    if verbose:
        print('Precision:  {0:1.4f}'.format(prec_))

    return prec_


def recall(predictions, binary_threshold=4, greater_than_threshold=True, zero_division=0, verbose=True):
    """Compute recall for thresholded predictions.

    .. math::
        \\text{recall} = \\frac{tp}{tp + fn}

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        binary_threshold
            threshold for binary classification
        greater_than_threshold
            If True, an estimated rating is considered positive iff est > binary_threshold,
            and a true rating is considered positive iff true_r > binary_threshold.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Recall of thresholded predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    tp_ = np.sum([(true_r > positive_threshold and est > positive_threshold)
                    for (_, _, true_r, est, _) in predictions])

    fn_ = np.sum([(true_r > positive_threshold and est <= positive_threshold)
                    for (_, _, true_r, est, _) in predictions])

    rec_ = tp_/(tp_ + fn_) if (tp_ + fn_) > 0 else zero_division

    if verbose:
        print('Recall:  {0:1.4f}'.format(rec_))

    return rec_


def f1score(predictions, binary_threshold=4, greater_than_threshold=True, zero_division=0, verbose=True):
    """Compute F1 score for thresholded predictions.

    .. math::
        \\text{f1score} = \\frac{2* precision * recall}{precision + recall}

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        binary_threshold
            threshold for binary classification
        greater_than_threshold
            If True, an estimated rating is considered positive iff est > binary_threshold,
            and a true rating is considered positive iff true_r > binary_threshold.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The F1 score of thresholded predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    prec_ = precision(predictions, binary_threshold, greater_than_threshold, verbose=False)

    rec_ = recall(predictions, binary_threshold, greater_than_threshold, verbose=False)

    f1_ = 2 * prec_ * rec_ / (prec_ + rec_) if (prec_ + rec_) > 0 else zero_division

    if verbose:
        print('F1 score:  {0:1.4f}'.format(f1_))

    return f1_
