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
    Index segmentation.
    :param n: Input array
    :param n_window: window size
    :param n_seg: segmentation size
    :return: start, end index list
    """
    st, ed = [], []
    if n_seg is not None and n_window is not None:
        return -1
    if n_seg is None and n_window is None:
        return -1
    if n_seg is not None and n_window is None:
        n_window = int(np.ceil(n / n_seg))
    if n_window is not None and n_seg is None:
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
    Piecewise Aggregate Approximation.
    :param X: Input array
    :param n_window: window size
    :param n_seg: segmentation size
    :return: ppa array
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
    Symbolic Aggregate approXimation.
    :param X: Input array
    :param n_alphabet: alphabet size (percentage. n/n_alphabet)
    :return: sax array
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
    Symbol basket Generation.
    :param X: Input array
    :param n_alphabet: alphabet size (percentage. n/n_alphabet)
    :return: basket id, basket
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
            items = np.concatenate((b_line, t_line)).tolist()

            basket_id.append(i)
            basket.append(items)

    return basket_id, basket


def symbol_sequence_baskets(X, s_id, n_alphabet):
    """
    symbol sequence basket generation
    :param X: sax array
    :param s_id: sequential id
    :param n_alphabet: alphabet size (percentage. n/n_alphabet)
    :return: symbol sequence basket
    """
    basket_id, basket = symbol_baskets(X, n_alphabet)

    s_basket = []
    for i in range(len(basket_id)):
        s_basket.append([s_id, basket_id[i], basket[i]])
    s_basket = np.array(s_basket)

    return s_basket


def flatten_unique(X):
    """
    Convert 2N array to 1N array.
    :param X: symbol sequence basket
    :return: flatten array
    """
    unique = []
    for line in X:
        unique = np.concatenate((unique, line[2]))

    unique = np.unique(unique)

    return unique


def idx_value_convert(X, idx):
    """
    Convert basket value to index
    :param X: symbol sequence basket
    :param idx: flatten array (symbol sequence basket)
    :return: converted array
    """
    res = []
    for i in range(len(X)):
        tmp = []
        for j in range(len(X[i][2])):
            tmp.append(np.where(idx == X[i][2][j])[0][0])
        res.append([X[i][0], X[i][1], tmp])

    return res


if __name__ == "__main__":
    pass
