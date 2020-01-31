import numpy as np


def otsu_threshold(score, ):
    """
    Otsu threshold
    :param score: anomaly score
    :return: threshold
    """
    score = np.array(score)
    cMax, thr = 0., 0.
    for t in np.arange(np.min(score), np.max(score), 0.01):
        clsL = score[np.where(score < t)]
        clsH = score[np.where(score >= t)]
        wL = clsL.size / score.size
        wH = clsH.size / score.size
        meanL = np.mean(float(wL))
        meanH = np.mean(float(wH))
        cVal = wL * wH * (meanL - meanH) ** 2.
        if cVal > cMax:
            cMax, thr = cVal, t

    return thr


def iqr_threshold(score, p_high=75, p_low=25, weight=1):
    """
    IQR threshold
    :param score: anomaly score
    :param p_high:
    :param p_low:
    :param weight:
    :return:
    """
    q3, q1 = np.percentile(score, [p_high, p_low])
    iqr = q3 - q1
    thr_high = (q3 + (iqr * 1.5)) * weight
    thr_low = (q1 - (iqr * 1.5)) * weight

    return thr_high, thr_low
