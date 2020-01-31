import sklearn.preprocessing as normalizer


def max_abs_scaler(X, copy=True):
    """
    Scale each feature by its maximum absolute value.
    :param X: numpy array. Input array.
    :param copy: boolean, optional, default True.
    :return: Max-Abs transformer
    """
    transformer = normalizer.MaxAbsScaler(copy=copy).fit(X)

    return transformer


def min_max_scaler(X, feature_range=(0, 1), copy=True):
    """
    Transform features by scaling each feature to a given range.
    :param X: numpy array. Input array.
    :param feature_range: Desired range of transformed data.
    :param copy: boolean, optional, default True.
    :return: Min-Max transformer
    """
    transformer = normalizer.MinMaxScaler(feature_range=feature_range, copy=copy).fit(X)

    return transformer


def normalize(X, method='l2', copy=True):
    """
    Normalize samples individually to unit norm.
    :param X: numpy array. Input array.
    :param method: ‘l1’, ‘l2’, or ‘max’, optional (‘l2’ by default). The norm to use to normalize each non zero sample.
    :param copy: boolean, optional, default True.
    :return: Normalize transformer
    """
    transformer = normalizer.Normalizer(norm=method, copy=copy).fit(X)

    return transformer


def robust_scaler(X, quantile_range=(25., 75.), centering=True, scaling=True, copy=True):
    """
    Scale features using statistics that are robust to outliers.
    :param X: numpy array. Input array.
    :param quantile_range: tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0.
                            Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR Quantile
                            range used to calculate scale_.
    :param centering: boolean, True by default.
    :param scaling: boolean, True by default. If True, scale the data to interquartile range.
    :param copy: boolean, optional, default True.
    :return: Robust transformer
    """
    transformer = normalizer.RobustScaler(quantile_range=quantile_range,
                                          with_centering=centering,
                                          with_scaling=scaling,
                                          copy=copy).fit(X)

    return transformer


def standard_scaler(X, mean=True, std=True, copy=True):
    """
    Standardize features by removing the mean and scaling to unit variance.
    :param X: numpy array. Input array
    :param mean: boolean, True by default.
    :param std: boolean, True by default. If True, scale the data to unit variance.
    :param copy: boolean, optional, default True.
    :return: Standard transformer
    """
    transformer = normalizer.StandardScaler(with_mean=mean, with_std=std, copy=copy).fit(X)

    return transformer


if __name__ == "__main__":
    import numpy as np

    print("Normalizer test.")
    print("Max Abs scaler test.")
    arr = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
    maxabs = max_abs_scaler(arr)
    print(maxabs.transform(arr))

    print("Min Max scaler test.")
    arr = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    minmax = min_max_scaler(arr)
    print(minmax.transform(arr))

    print("Normalize scaler test.")
    arr = np.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
    norm = normalize(arr)
    print(norm.transform(arr))

    print("Robust scaler test.")
    arr = np.array([[1., -2., 2.], [-2., 1., 3.], [4., 1., -2.]])
    robust = robust_scaler(arr)
    print(robust.transform(arr))

    print("Standard scaler test.")
    arr = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    standard = standard_scaler(arr)
    print(standard.transform(arr))
