# -*- coding: utf-8 -*-
from RDA.statistics import mspc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":
    acc_data = pd.read_csv('samples/accelerometer.csv')
    acc_columns = acc_data.columns.tolist()
    acc_features = acc_columns[:-1]
    n_acc_features = len(acc_features)
    train = acc_data[acc_features]
    labels = acc_data[acc_columns[n_acc_features]]

    true = labels[labels == 0]
    false = labels[labels == 1]

    all_score, _, lcl, ucl = mspc.tsquare_single(train, n_acc_features)
    scores = mspc.tsquare_decomposition(train, acc_features)
    sub = mspc.tsquare_residual(all_score, scores)

    idx = false.tolist()

    vals = []
    for i in range(n_acc_features):
        tmp = [sub[i][j] for j in idx]
        vals.append(tmp)

    for i in range(len(vals)):
        print(acc_features[i], np.sum(vals[i]))


