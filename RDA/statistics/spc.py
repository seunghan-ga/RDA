# -*- coding: utf-8 -*-
from RDA.statistics.tables import A2, A3, B3, B4, D3, D4
import numpy as np


def _c(data, size=1, lcl=None, ucl=None):
    _type = "C"

    sizes, data = data.T
    if (size - 1) == 1:
        sizes, data = data, sizes

    assert np.mean(sizes) == sizes[0]

    cbar = np.mean(data)

    if lcl is None and ucl is None:
        lcl = cbar - 3 * np.sqrt(cbar)
        ucl = cbar + 3 * np.sqrt(cbar)

    return data, cbar, lcl, ucl, _type


def _u(data, size=1, lcl=None, ucl=None, newdata=None):
    _type = "U"

    sizes, data = data.T
    if (size - 1) == 1:
        sizes, data = data, sizes

    data2 = data / sizes
    ubar = np.sum(data) / np.sum(sizes)

    lcl_list, ucl_list = [], []
    if lcl is None and ucl is None:
        for i in sizes:
            lcl_list.append(ubar - 3 * np.sqrt(ubar / i))
            ucl_list.append(ubar + 3 * np.sqrt(ubar / i))
    else:
        for i in sizes:
            lcl_list.append(lcl)
            ucl_list.append(ucl)

    return data2, ubar, lcl_list, ucl_list, _type


def _p(data, size=1, lcl=None, ucl=None, newdata=None):
    _type = "P"

    sizes, data = data.T
    if (size - 1) == 1:
        sizes, data = data, sizes

    data2 = data / sizes
    pbar = np.mean(data2)

    for n in sizes:
        assert n * pbar >= 5
        assert n * (1 - pbar) >= 5

    if np.mean(sizes) == sizes[0]:
        size = sizes[0]
        if lcl is None and ucl is None:
            lcl = pbar - 3 * np.sqrt((pbar * (1 - pbar)) / size)
            ucl = pbar + 3 * np.sqrt((pbar * (1 - pbar)) / size)

            if lcl < 0:
                lcl = 0
            if ucl > 1:
                ucl = 1

        return data2, pbar, lcl, ucl, _type

    else:
        lcl_list, ucl_list = [], []
        if lcl is None and ucl is None:
            for size in sizes:
                lcl_list.append(pbar - 3 * np.sqrt((pbar * (1 - pbar)) / size))
                ucl_list.append(pbar + 3 * np.sqrt((pbar * (1 - pbar)) / size))
        else:
            for size in sizes:
                lcl_list.append(lcl)
                ucl_list.append(ucl)

        return data2, pbar, lcl, ucl, _type


def _np(data, size=1, lcl=None, ucl=None, newdata=None):
    _type = "NP"

    sizes, data = data.T
    if (size - 1) == 1:
        sizes, data = data, sizes

    assert np.mean(sizes) == sizes[0]

    p = np.mean([float(d) / sizes[0] for d in data])
    pbar = np.mean(data)

    if lcl is None and ucl is None:
        lcl = pbar - 3 * np.sqrt(pbar * (1 - p))
        ucl = pbar + 3 * np.sqrt(pbar * (1 - p))

    return data, pbar, lcl, ucl, _type


def _xbar_r(data, size, lcl=None, ucl=None, newdata=None):
    _type = "Xbar-R"

    assert size >= 2
    assert size <= 10
    newvalues = None

    R, X = [], []  # values
    for xs in data:
        assert len(xs) == size
        R.append(max(xs) - min(xs))
        X.append(np.mean(xs))

    if newdata:
        newvalues = [np.mean(xs) for xs in newdata]

    Rbar = np.mean(R)  # center
    Xbar = np.mean(X)

    if lcl is None and ucl is None:
        lcl = Xbar - A2[size] * Rbar
        ucl = Xbar + A2[size] * Rbar

    return X, Xbar, lcl, ucl, _type


def _rbar(data, size, lcl=None, ucl=None, newdata=None):
    _type = "R"

    assert size >= 2
    assert size <= 10
    newvalues = None

    R = []  # values
    for xs in data:
        assert len(xs) == size
        R.append(max(xs) - min(xs))

    if newdata:
        newvalues = [max(xs) - min(xs) for xs in newdata]

    Rbar = np.mean(R)  # center

    if lcl is None and ucl is None:
        lcl = D3[size] * Rbar
        ucl = D4[size] * Rbar

    return R, Rbar, lcl, ucl, _type


def _xbar_s(data, size, lcl=None, ucl=None, newdata=None):
    _type = "Xbar-S"

    assert size >= 2
    assert size <= 10
    newvalues = None

    X, S = [], []
    for xs in data:
        assert len(xs) == size
        S.append(np.std(xs, ddof=1))
        X.append(np.mean(xs))

    if newdata:
        newvalues = [np.mean(xs) for xs in newdata]

    sbar = np.mean(S)
    xbar = np.mean(X)

    if lcl is None and ucl is None:
        lclx = xbar - A3[size] * sbar
        uclx = xbar + A3[size] * sbar
    else:
        lclx = lcl
        uclx = ucl

    return X, xbar, lclx, uclx, _type


def _sbar(data, size, lcl=None, ucl=None, newdata=None):
    _type = "S"

    assert size >= 2
    assert size <= 10
    newvalues = None

    S = []
    for xs in data:
        assert len(xs) == size
        S.append(np.std(xs, ddof=1))

    if newdata:
        newvalues = [np.std(xs, ddof=1) for xs in newdata]

    sbar = np.mean(S)

    if lcl is None and ucl is None:
        lcls = B3[size] * sbar
        ucls = B4[size] * sbar
    else:
        lcls = lcl
        ucls = ucl

    return S, sbar, lcls, ucls, _type
