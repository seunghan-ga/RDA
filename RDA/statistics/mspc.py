# -*- coding: utf-8 -*-
from scipy.stats import beta
import numpy as np


def tsquare_single(data, size, lcl=None, ucl=None, center=None):
    """
    Calculate Hotelling's T2 score
    :param data: Input array.
    :param size: Number of columns
    :param lcl: Lower control limit value
    :param ucl: Upper control limit value
    :param center: center value
    :return: T2 score, center, lcl, ucl
    """
    data = np.array(data)
    numsample = len(data)

    colmean = np.mean(data, axis=0)
    matcov = np.cov(data.T)
    matinv = np.linalg.inv(matcov)

    values = []
    for sample in data:
        dif = sample - colmean
        value = dif.T @ matinv @ dif
        values.append(value)

    cl = ((numsample - 1) ** 2) / numsample

    if lcl is None:
        lcl = cl * beta.ppf(0.00135, size / 2, (numsample - size - 1) / 2)
    if ucl is None:
        ucl = cl * beta.ppf(0.99865, size / 2, (numsample - size - 1) / 2)
    if center is None:
        center = cl * beta.ppf(0.5, size / 2, (numsample - size - 1) / 2)

    return np.array(values), center, lcl, ucl


def tsquare_decomposition(data, columns):
    """
    Decompose T2 score
    :param data: Input array
    :param columns: t2 columns
    :return: Decomposed T2 scores
    """
    t2_scores = []
    for col in columns:
        cols = columns[:]
        cols.remove(col)
        score, _, _, _ = tsquare_single(data[cols], len(cols))
        t2_scores.append(score)

    return np.array(t2_scores)


def tsquare_residual(all_score, decompose_score):
    """
    Calculate residual
    :param all_score: All variable score
    :param decompose_score: decomposed score
    :return: residual array
    """
    residual = []
    for score in decompose_score:
        residual.append(all_score - score)

    return np.array(residual)


def ano_score_idx(t2_score, lcl=None, ucl=None):
    """
    Anomaly detection
    :param t2_score: T2 score array
    :param lcl: Lower control limit value
    :param ucl: Upper control limit value
    :return: Anomaly point index
    """
    ano_idx = []
    for idx in range(len(t2_score)):
        if ucl < t2_score[idx]:
            ano_idx.append(idx)

    return ano_idx


def ewma(data, size, target=None, weight=0.2):
    assert ((weight > 0) and (weight < 1))

    if size > 1:
        data = np.mean(data, axis=1)

    if target is None:
        target = np.mean(data)

    rbar = []
    for i in range(data.__len__() - 1):
        rbar.append(abs(data[i] - data[i + 1]))
    std = np.mean(rbar) / 1.128

    ewma = []
    i = target
    for x in data:
        ewma.append(weight * x + (1 - weight) * i)
        i = ewma[-1]

    lcl, ucl = [], []
    for i in range(1, data.__len__() + 1):
        lcl.append(target - 3 * (std) * np.sqrt((weight / (2 - weight)) * (1 - (1 - weight) ** (2 * i))))
        ucl.append(target + 3 * (std) * np.sqrt((weight / (2 - weight)) * (1 - (1 - weight) ** (2 * i))))

    return np.array(ewma), target, lcl, ucl
