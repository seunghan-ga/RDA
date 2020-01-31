# -*- coding: utf-8 -*-
from scipy.stats import beta, f, chisquare
import numpy as np


def tsquare_single(data, size, lcl=None, ucl=None, center=None, newdata=None):
    _title = "T-square Single Chart"

    data = np.array(data)
    numsample = data.__len__()

    colmean = np.mean(data, axis=0)
    matcov = np.cov(data.T)
    matinv = np.linalg.inv(matcov)

    values = []
    for sample in data:
        dif = sample - colmean
        value = matinv.dot(dif.T).dot(dif)
        values.append(value)

    cl = ((numsample - 1) ** 2) / numsample

    if lcl is None:
        lcl = cl * beta.ppf(0.00135, size / 2, (numsample - size - 1) / 2)
    if ucl is None:
        ucl = cl * beta.ppf(0.99865, size / 2, (numsample - size - 1) / 2)
    if center is None:
        center = cl * beta.ppf(0.5, size / 2, (numsample - size - 1) / 2)

    return values, center, lcl, ucl, _title


# test
def tsquare_double(data, size, newdata=None):
    _title = "T-square Double Chart"

    sizes = data[:, 0]
    sample = data[:, 1:]

    samples = dict()
    for n, value in zip(sizes, sample):
        if n in samples:
            samples[n] = np.vstack([samples[n], value])
        else:
            samples[n] = value

    m = len(samples.keys())
    n = len(samples[1])
    p = len(samples[1].T)

    variance, S = [], []
    for i in range(m):
        mat = np.cov(samples[i + 1].T, ddof=1)
        variance.append(mat.diagonal())
        for j in range(1, mat.shape[0]):
            cov = np.hstack((np.array([]), mat.diagonal(i)))
        S.append(cov)

    variance, S = np.array(variance), np.array(S)

    means = np.array([samples[xs + 1].mean(axis=0) for xs in range(m)])
    means_total = means.mean(axis=0)

    varmean = variance.mean(axis=0)
    smean = S.mean(axis=0)
    n = len(varmean)
    mat = np.zeros(shape=(n, n)) + np.diag(varmean)

    a, b = 0, n - 1
    for i in range(1, n):
        mat = mat + np.diag(smean[a:b], k=i) + np.diag(smean[a:b], k=-i)
        a, b = b, (b + (b - 1))

    Smat = mat
    Smat_inv = np.linalg.inv(Smat)

    values = []
    for i in range(m):
        a = means[i] - means_total
        values.append(5 * a @ Smat_inv @ a.T)

    p1 = (p * (m - 1) * (n - 1))
    p2 = (m * n - m - p + 1)
    lcl = (p1 / p2) * f.ppf(0.00135, p, p2)
    center = (p1 / p2) * f.ppf(0.50, p, p2)
    ucl = (p1 / p2) * f.ppf(0.99865, p, p2)

    return values, center, lcl, ucl, _title


def ewma(data, size, target=None, weight=0.2, newdata=None):
    _title = "EWMA Chart"

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

    return ewma, target, lcl, ucl, _title
