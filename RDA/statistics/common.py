# -*- coding: utf-8 -*-
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


def computePCA(df, features, n=2):
    # Separating out the features
    data = df.loc[:, features].values

    # Standardizing the features
    data = StandardScaler().fit_transform(data)

    # Create columns
    cols = []
    for i in range(n):
        col = "PC_%d" % (i + 1)
        cols.append(col)

    # Compute PCA
    pca = PCA(n_components=n)
    transformed = pca.fit_transform(data)
    score = pd.DataFrame(data=transformed, columns=cols)

    return score


# ing 2019-06-10
def computeLDA(df, features, n=2):
    X = df.values
    y = []
    for i in range(len(X)):
        # y.append()
        pass

    print(X)
    lda = LinearDiscriminantAnalysis(n_components=n)

    X_r2 = lda.fit(X, y).transform(X)

    return X_r2


if __name__ == "__main__":
    pass