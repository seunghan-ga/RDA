# -*- coding: utf-8 -*-s
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
import numpy as np
import pandas as pd


class anoLSTM(object):
    def __init__(self):
        pass

    def split_sequences(self, sequences, n_in, n_out):
        """
        데이터 변환
        :param sequences: 분석에 필요한 시퀀스 데이터
        :param n_in: 윈도우 크기
        :param n_out: 예측 범위
        :return: 윈도우 크기에 맞게 변환된 데이터
        """
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_in
            out_end_ix = end_ix + n_out

            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break

            # gather input and output parts of the pattern
            seq_x = sequences[i:end_ix, :]
            X.append(seq_x)

        return np.array(X)

    def data_preparation(self, data, scaler, n_in, n_out):
        """
        입력 데이터 표준화
        :param data: 표준화 할 데이터
        :param train_data: 표준화 기준이 되는 데이터
        :param n_in: 윈도우 크기
        :param n_out: 예측 범위
        :return: 표준화된 데이터, 표준화 객체
        """
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(data)

        data_scaled = scaler.transform(data)
        data_3d = self.split_sequences(data_scaled, n_in, n_out)

        return data_3d, scaler

    def lstm_autoencoder(self, train_data, n_in=1, hidden_dim=1, activation='tanh', bias_initializer='zeros',
                         kernel_initializer='he_normal', unit_forget_bias=True):
        """
        모델 구축
        :param train_data: 학습 데이터
        :param n_in: 윈도우 크기
        :param hidden_dim: 히든레이어의 유닛수
        :return: lstm_autoencoder 모형
        """
        ## encoder
        visible = tf.keras.layers.Input(shape=(n_in, train_data.shape[2]))
        encoder = tf.keras.layers.LSTM(hidden_dim, activation=activation, bias_initializer=bias_initializer,
                                       unit_forget_bias=unit_forget_bias, kernel_initializer=kernel_initializer,
                                       return_sequences=False)(visible)

        ## decoder
        decoder = tf.keras.layers.RepeatVector(n_in)(encoder)
        decoder = tf.keras.layers.LSTM(hidden_dim, activation=activation, return_sequences=True,
                                       bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias,
                                       kernel_initializer=kernel_initializer)(decoder)
        decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(train_data.shape[2],
                                                                        activation=activation))(decoder)

        ## model compile
        model = tf.keras.models.Model(inputs=visible, outputs=decoder)
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

        return model

    def flat(self, X):
        """
        3D array -> 2D array
        :param X : 3D array
        :param Return : 2D array
        """
        flattened_X = np.empty((X.shape[0], X.shape[2]))

        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]

        return flattened_X

    def prediction(self, model, scaler, true_data):
        """
        예측
        :param model: 구축한 모델
        :param true_data: 예측값과 비교하는 실제 데이터
        :param scaler: 표준화 객체
        :return: 실제값과 예측값
        """

        pred_v3d = model.predict(true_data)
        pred_v = self.flat(pred_v3d)
        true_v = self.flat(true_data)

        # 역변환
        true_v = pd.DataFrame(scaler.inverse_transform(true_v))
        pred_v = pd.DataFrame(scaler.inverse_transform(pred_v))

        return true_v, pred_v

    def labeling(self, data, n_feature):
        """
        실제 데이터에 라벨링 붙이기
        :param data: 라벨이 없는 데이터
        :param n_feature: 데이터의 변수 개수
        :return: 실제값 라벨
        """
        # 이상치 판단 기준 : (mu-3sigma , mu + 3sigma)
        mu = data.mean()
        sd = data.std()

        upper = mu + 3 * sd
        lower = mu - 3 * sd

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
        # 잔차
        error_v = np.abs(true - pred)

        # 평균 공분산 추정
        e_mu = np.array(np.mean(error_v))
        e_cov = np.array(np.cov(error_v.T))

        # 이상치 점수 구하기
        score = np.array(
            [np.transpose((np.array(error_v)[i] - e_mu)) @ np.linalg.inv(e_cov) @ (np.array(error_v)[i] - e_mu) for i in
             range(error_v.shape[0])])

        return score

    def find_criteria(self, true, pred, label_validation):
        """
        f1 점수가 최대화 되는 점 구하기
        :param true: 실제 데이터
        :param pred: 예측 데이터
        :param label_validation: 실제 데이터에 대한 라벨
        :return: 이상치 기준값
        """
        # 이상치 점수 구하기
        score = self.anomaly_score(true, pred)
        thr = np.arange(1, np.max(score))
        f1 = list()
        recall = list()
        precision = list()

        # 기준값에 따른 f1 점수 구하기
        for j in range(len(thr)):
            pred_label = list()
            for i in range(len(score)):
                if score[i] > thr[j]:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            tn, fp, fn, tp = confusion_matrix(label_validation, pred_label).ravel()
            recall.append(tp / (tp + fn))
            precision.append(tp / (tp + fp))
            f1.append(f1_score(label_validation, pred_label, average='binary'))

        # f1 점수가 가장 큰 기준값 선정
        selected_thr = thr[np.where(np.max(f1) == f1)][0]
        return selected_thr

    def f1score(self, anomaly_score, label_test, threshold):
        """
        선택된 기준값으로 f1점수 구하기
        :param anomaly_score: 이상치 점수
        :param label_test: 실제 데이터에 대한 라벨
        :param threshold: 이상치  기준값
        :return: f1 점수, 예측값에 대한 라벨
        """
        pred_test = []
        ab_idx = []
        for i in range(len(anomaly_score)):
            if anomaly_score[i] > threshold:
                pred_test.append(1)
                ab_idx.append(i)
            else:
                pred_test.append(0)
        tn, fp, fn, tp = confusion_matrix(label_test, pred_test).ravel()
        f1_test = f1_score(label_test, pred_test, average='binary')
        pred_test = pd.Series(pred_test)

        return f1_test, pred_test, ab_idx

    def MSE(self, true, pred, pred_label_test, feature):
        """
        이상치 구간에서 영향을 미치는 변수 확인
        :param true: 실제값
        :param pred: 예측값
        :param pred_label_test: 예측값에 대한 라벨
        :param feature: 변수명
        :return: mse값
        """
        point = np.where(pred_label_test == 1)[0]
        mse = list()
        for p in point:
            mse.append(np.array(np.power(true.loc[p] - pred.loc[p], 2)))

        mse = pd.DataFrame(mse)
        mse.columns = feature

        return mse

    def outlier(self, true, pred, var):
        """
        :param true : 실제값
        :param pred : 예측값
        :param var :  변수명
        :return : 잔차, 이상치 위치(잔차)
        """
        residual = (true[var] - pred[var]).values.tolist()
        mean = np.mean(residual)
        std = np.std(residual)

        lcl = mean - std * 3
        ucl = mean + std * 3

        ab_idx = []
        for i in range(len(residual)):
            if residual[i] >= ucl or residual[i] <= lcl:
                ab_idx.append(i)

        return residual, ab_idx

    def outlier2(self, residual, point_3sd, var, thr=0):
        """
        :param residual : 잔차
        :param point_3sd : 이상치 위치(statistics-3sigma)
        :param var :  변수명
        :param thr : 이상치 기준값
        :return : 잔차, 누적잔차, 이상치 기준값
        """
        size = len(residual)
        cumulative = []
        ab_idx = []

        if thr == 0:
            thr = [(np.mean(residual) - np.std(residual) * 3), (np.mean(residual) + np.std(residual) * 3)]

        for i in range(size):
            if i > 0 and i in point_3sd:
                cumulative.append(0)
                continue
            if i == 0:
                cumulative.append(residual[i])
            else:
                cumulative.append(cumulative[i - 1] + residual[i])

        for i in point_3sd:
            if np.abs(cumulative[i]) >= np.abs(residual[i]):
                ab_idx.append(i)

        return residual, cumulative, thr

    def computeSPC(self, true, feature, spc_func):
        """
        :param true : 실제값
        :param feature: 컬럼명
        :param spc_func : spc함수
        :return : statistics 결과
        """
        _type = spc_func._type
        # P
        if _type.lower() == ("default").lower():
            data = pd.DataFrame()
            control_limit = {}
            for col in feature:
                res = spc_func.default(true[col].values)
                points, center, lcl, ucl = res[0], res[1], res[2], res[3]
                data[col] = pd.Series(points, true.index)
                control_limit[col] = [center, lcl, ucl]

            return data, control_limit

        if _type.lower() == ("P").lower():
            true['n'] = pd.Series(len(true), true.index)
            data = pd.DataFrame()
            control_limit = {}
            for col in feature:
                res = spc_func.p(true[['n', col]].values)
                points, center, lcl, ucl = res[0], res[1], res[2], res[3]
                data[col] = pd.Series(points, true.index)
                control_limit[col] = [center, lcl, ucl]

            return data, control_limit

        if _type.lower() == ("NP").lower():
            true['n'] = pd.Series(len(true), true.index)
            data = pd.DataFrame()
            control_limit = {}
            for col in feature:
                res = spc_func.np(true[['n', col]].values)
                points, center, lcl, ucl = res[0], res[1], res[2], res[3]
                data[col] = pd.Series(points, true.index)
                control_limit[col] = [center, lcl, ucl]

            return data, control_limit

        if _type.lower() == ("C").lower():
            true['n'] = pd.Series(len(true), true.index)
            data = pd.DataFrame()
            control_limit = {}
            for col in feature:
                res = spc_func.c(true[['n', col]].values)
                points, center, lcl, ucl = res[0], res[1], res[2], res[3]
                data[col] = pd.Series(points, true.index)
                control_limit[col] = [center, lcl, ucl]

            return data, control_limit

        if _type.lower() == ("U").lower():
            true['n'] = pd.Series(len(true), true.index)
            data = pd.DataFrame()
            control_limit = {}
            for col in feature:
                res = spc_func.u(true[['n', col]].values)
                points, center, lcl, ucl = res[0], res[1], res[2], res[3]
                data[col] = pd.Series(points, true.index)
                control_limit[col] = [center, lcl, ucl]

            return data, control_limit


if __name__ == "__main__":
    pass