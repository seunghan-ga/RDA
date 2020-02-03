# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
import pandas as pd


class anoLSTM(object):
    def __init__(self):
        pass

     # def f1score(self, anomaly_score, label_test, threshold):
    #     """
    #     선택된 기준값으로 f1점수 구하기
    #     :param anomaly_score: 이상치 점수
    #     :param label_test: 실제 데이터에 대한 라벨
    #     :param threshold: 이상치  기준값
    #     :return: f1 점수, 예측값에 대한 라벨
    #     """
    #     pred_test = []
    #     ab_idx = []
    #     for i in range(len(anomaly_score)):
    #         if anomaly_score[i] > threshold:
    #             pred_test.append(1)
    #             ab_idx.append(i)
    #         else:
    #             pred_test.append(0)
    #
    #     f1_test = f1_score(label_test, pred_test, average='binary')
    #     pred_test = pd.Series(pred_test)
    #
    #     return f1_test, pred_test, ab_idx

    # def outlier(self, true, pred, var):
    #     """
    #     :param true : 실제값
    #     :param pred : 예측값
    #     :param var :  변수명
    #     :return : 잔차, 이상치 위치(잔차)
    #     """
    #     residual = (true[var] - pred[var]).values.tolist()
    #     mean = np.mean(residual)
    #     std = np.std(residual)
    #
    #     lcl = mean - std * 3
    #     ucl = mean + std * 3
    #
    #     ab_idx = []
    #     for i in range(len(residual)):
    #         if residual[i] >= ucl or residual[i] <= lcl:
    #             ab_idx.append(i)
    #
    #     return residual, ab_idx

    # def outlier2(self, residual, point_3sd, var, thr=0):
    #     """
    #     :param residual : 잔차
    #     :param point_3sd : 이상치 위치(statistics-3sigma)
    #     :param var :  변수명
    #     :param thr : 이상치 기준값
    #     :return : 잔차, 누적잔차, 이상치 기준값
    #     """
    #     size = len(residual)
    #     cumulative = []
    #     ab_idx = []
    #
    #     if thr == 0:
    #         thr = [(np.mean(residual) - np.std(residual) * 3), (np.mean(residual) + np.std(residual) * 3)]
    #
    #     for i in range(size):
    #         if i > 0 and i in point_3sd:
    #             cumulative.append(0)
    #             continue
    #         if i == 0:
    #             cumulative.append(residual[i])
    #         else:
    #             cumulative.append(cumulative[i - 1] + residual[i])
    #
    #     for i in point_3sd:
    #         if np.abs(cumulative[i]) >= np.abs(residual[i]):
    #             ab_idx.append(i)
    #
    #     return residual, cumulative, thr

# =====================================================================================================================

    def flatten(self, X):
        """
        Flatten a 3D array.
        :param X: A 3D array for lstm, where the array is sample x timesteps x features.
        :return: A 2D array, sample x features.
        """
        # sample x features array.
        flattened_X = np.empty((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]

        return np.array(flattened_X)

    def scaling(self, X, scaler):
        """
        Scaling 3D array.
        :param X: A 3D array for lstm, where the array is sample x timesteps x features.
        :param scaler: A scaler object, e.g., StandardScaler, normalize ...
        :return: Scaled 3D array.
        """
        for i in range(X.shape[0]):
            X[i, :, :] = scaler.transform(X[i, :, :])

        return X

    def temporalize(self, X, y, lookback):
        """
        Slice data
        :param X:
        :param y:
        :param lookback:
        :return: Temporalized data
        """
        output_X = []
        output_y = []
        for i in range(len(X) - lookback - 1):
            t = []
            for j in range(1, lookback + 1):  # Gather past records upto the lookback period
                t.append(X[[(i + j + 1)], :])
            output_X.append(t)
            output_y.append(y[i + lookback + 1])

        return output_X, output_y

    def preparation(self, X=None, y=None, timesteps=1, n_features=1):
        """
        convert data (2d > 3d)
        :param X: 2D Raw data
        :param y: 2D Raw data labels. (default : None)
        :param timesteps: Window size
        :param n_features: Column length
        :return: 3D array
        """
        if y is None:
            input_y = np.zeros(len(X))
        else:
            input_y = y

        X, y = self.temporalize(X=X, y=input_y, lookback=timesteps)
        X = np.array(X)
        X_3D = X.reshape((X.shape[0], timesteps, n_features))

        return X_3D

    def get_model(self, timesteps=1, n_features=1, **kwargs):
        """
        Get LSTM Autoencoder model
        :param timesteps: Window size
        :param n_features: Column length
        :param **kwargs: Parameters
        :return: LSTM model
        """
        # set parameter
        units = kwargs['units']
        activate_fn = kwargs['activate_func']  # default: tanh
        recurrent_fn = kwargs['recurrent_func']
        optimize_fn = kwargs['optimize_func']
        loss_fn = kwargs['loss_func']
        kernel_init = kwargs['kernel_init']  # default: glorot_uniform
        dropout = kwargs['dropout']

        LSTM_Model = tf.keras.Sequential()
        # encoder
        LSTM_Model.add(tf.keras.layers.LSTM(units, activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=True, dropout=dropout,
                                            input_shape=(timesteps, n_features)))
        LSTM_Model.add(tf.keras.layers.LSTM(int(units / 2), activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=False, dropout=dropout))
        LSTM_Model.add(tf.keras.layers.RepeatVector(timesteps))
        # decoder
        LSTM_Model.add(tf.keras.layers.LSTM(int(units / 2), activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=True, dropout=dropout))
        LSTM_Model.add(tf.keras.layers.LSTM(units, activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=True, dropout=dropout))
        LSTM_Model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))
        LSTM_Model.compile(optimizer=optimize_fn, loss=loss_fn)

        return LSTM_Model

    def contribution(self, true, pred, pred_label, features):
        """
        Contribution of variable
        :param true: true data
        :param pred: predict data
        :param pred_label: predict data label
        :param features: columns
        :return: contribution
        """
        # df_true = pd.DataFrame(true)
        # df_pred = pd.DataFrame(pred)

        point = np.where(pred_label == 1)[0]
        mse = []
        for i in range(len(point)):
            if pred_label[i] == 1:
                # mse.append(np.array(np.power(df_true.loc[i] - df_pred.loc[i], 2)))
                mse.append(np.array(np.power(true[i] - pred[i], 2)))
            else:
                mse.append(np.zeros(len(features)))

        c_vals = pd.DataFrame(mse)
        c_vals.columns = features

        return c_vals

    def labeling(self, data, n_feature):
        """
        실제 데이터에 라벨링 붙이기
        :param data: no label data
        :param n_feature: columns
        :return: labeled data
        """
        upper = np.mean(data) + 3 * np.std(data)
        lower = np.mean(data) - 3 * np.std(data)

        label_list = []
        for i in range(data.shape[0]):
            if i == data.shape[0]:
                break

            condition = (np.array(lower) < np.array(data.iloc[i:i + 1, :])) & (
                    np.array(upper) > np.array(data.iloc[i:i + 1, :]))

            if np.sum(condition) == n_feature:
                label_list.append(0)
            else:
                label_list.append(1)

        label_v = pd.Series(label_list)

        return label_v

    def anomaly_score(self, true, pred):
        """
        이상치 점수 구하기
        :param true: 실제 데이터
        :param pred: 예측 데이터
        :return: 이상치 점수
        """
        error_v = np.abs(true - pred)  # u
        e_mu = np.mean(error_v)  # v
        e_cov = np.cov(error_v.T)  # V

        score = np.empty(error_v.shape[0])
        for i in range(error_v.shape[0]):
            score[i] = (np.sqrt((error_v[i] - e_mu) @ np.linalg.inv(e_cov) @ (error_v[i] - e_mu).T))

        return score

    def find_criteria(self, true, pred, label_validation):
        """
        Fine threshold.
        :param true: Input(real) data
        :param pred: Predicted data
        :param label_validation: real data label
        :return: threshold
        """
        # 이상치 점수 구하기
        score = self.anomaly_score(true, pred)
        thr = np.arange(1, np.max(score))
        f1 = []

        # 기준값에 따른 f1 점수 구하기
        for j in range(len(thr)):
            pred_label = []
            for i in range(len(score)):
                if score[i] > thr[j]:
                    pred_label.append(1)
                else:
                    pred_label.append(0)

            f1.append(f1_score(label_validation, pred_label, average='binary'))

        # f1 점수가 가장 큰 기준값 선정
        selected_thr = thr[np.where(np.max(f1) == f1)][0]

        return selected_thr


if __name__ == "__main__":
    pass