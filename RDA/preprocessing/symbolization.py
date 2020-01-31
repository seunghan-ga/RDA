# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np


def z_score_norm(X):
    """
    z_score normalization.
    :param X: Input array.
    :return: Normalized array.
    """
    s = np.std(X)
    x_bar = np.mean(X)
    x_norm = (X - x_bar) / s

    return np.array(x_norm)


def index_segmentation(n, n_seg=None, n_window=None):
    """

    :param n:
    :param n_window:
    :param n_seg:
    :return:
    """
    st, ed = [], []
    if n_window is None:
        n_window = int(np.ceil(n / n_seg))
    if n_seg is None:
        n_seg = int(np.ceil(n / n_window))
    n_last = int(n % n_seg)
    for i in range(n_seg):
        st.append(i * n_window)
        ed.append(i * n_window + n_window)
    if n_last > 0:
        ed[-1] = n

    return st, ed


def paa(X, n_seg=None, n_window=None):
    """

    :param X:
    :param n_window:
    :param n_seg:
    :return:
    """
    n_variable, n_timeseries = X.T.shape
    st, ed = None, None
    if n_window is not None:
        st, ed = index_segmentation(n_timeseries, n_window=n_window)
    elif n_seg is not None:
        st, ed = index_segmentation(n_timeseries, n_seg=n_seg)
    _paa = np.empty((n_variable, n_timeseries))

    t = tqdm(range(n_variable))
    for i in t:
        t.set_description("PAA progress (%s/%s)" % (i+1, len(t)))
        for j in range(len(st)):
            _paa[i][st[j]:ed[j]] = np.mean(X.T[i][st[j]:ed[j]])

    return _paa.T


def sax(X, n_alphabet=3):
    """

    :param X:
    :param n_alphabet:
    :return:
    """
    per = [(100/n_alphabet) * i for i in range(n_alphabet)]

    symbol = np.empty(shape=X.T.shape, dtype=np.str)
    symbol[:] = chr(97)

    t = tqdm(range(symbol.shape[0]))
    for i in t:
        t.set_description("SAX progress (%s/%s)" % (i+1, len(t)))
        for j in range(1, len(per)):
            _per = np.percentile(X.T[i], per)
            idx = np.where(X.T[i] >= _per[j])
            symbol[i][idx] = chr(97 + j)

    return symbol.T


def symbol_baskets(X, n_alphabet):
    """

    :param X:
    :param n_alphabet:
    :return:
    """
    lower, upper = chr(97), chr(97+(n_alphabet-1))
    basket = []
    basket_id = []
    t = tqdm(range(X.shape[0]))
    for i in t:
        t.set_description("Symbol baskets Generation progress (%s/%s)" % (i+1, len(t)))
        line = X[i]

        if lower in line or upper in line:
            idx_t = np.where(line == upper)[0]
            idx_b = np.where(line == lower)[0]
            t_line = ["%s.%s" % (ti, line[ti]) for ti in idx_t]
            b_line = ["%s.%s" % (bi, line[bi]) for bi in idx_b]

            basket_id.append(i)
            basket.append(np.concatenate((t_line, b_line)).tolist())

    return np.array(basket_id), np.array(basket)


def symbol_sequence_baskets(X, n_alphabet):
    pass


if __name__ == "__main__":
    pass
