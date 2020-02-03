# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
import numpy as np


def pca_transform(X, n=2):
    """
    Principal component analysis. (PCA)
    :param X: Input array.
    :param n: Number of PC.
    :return: PC transformer.
    """
    data = np.array(X)
    pca = PCA(n_components=n)
    transformer = pca.fit(data)

    return transformer


if __name__ == "__main__":
    pass