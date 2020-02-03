from RDA.analysis.anoLSTM import anoLSTM
import RDA.analysis.thrsehold as thr
import tensorflow as tf
import pandas as pd
import numpy as np
import pymysql
import joblib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)

    db_info = {
        "host": "192.168.1.101",
        "user": "hadoop",
        "password": "hadoop",
        "db": "DEMO",
        "charset": "utf8"
    }
    conn = pymysql.connect(**db_info)
    cursor = conn.cursor()
    cursor.execute("select * from sar order by date desc limit 100")
    columns = ["date", "type", "user", "nice", "system", "iowait", "steal", "idel"]
    data = pd.DataFrame(cursor.fetchall(), columns=columns)
    data = data[columns[2:]].astype("float").values
    # print(np.array(cursor.fetchall()))

    # parameter
    timesteps = 30  # window size
    features = columns[1:]
    n_features = 6
    units = len(features) * 2
    epochs = 1000
    batch_size = 32

    with tf.device('/GPU:0'):
        # scaler load
        scaler = joblib.load("RDA/models/normalization.scaler")
        print('Loaded scaler from disk')

        # model load
        json_file = open("RDA/models/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("RDA/models/model.h5")
        LSTM_Model = loaded_model
        print('Loaded model from disk')

        # adjust window size
        zeros = np.zeros((timesteps+1, n_features))
        test = np.concatenate((zeros, data), axis=0)

        # 3d transformation & scaling
        valid = anoLSTM().preparation(X=test, timesteps=timesteps, n_features=n_features)
        scaled_valid = anoLSTM().scaling(valid, scaler)

        # prediction
        pred_valid = LSTM_Model.predict(scaled_valid, verbose=0)

        # data flatten
        flatten_scaled_valid = anoLSTM().flatten(scaled_valid)
        flatten_pred_valid = anoLSTM().flatten(pred_valid)

        # scoring (mean square error)
        mse = np.mean(np.power(flatten_scaled_valid - flatten_pred_valid, 2), axis=1)
        error_df = pd.DataFrame({'Reconstruction_error': mse})
        err_values = error_df.Reconstruction_error.values

        # scoring (mahalanobis distance)
        ano_score = anoLSTM().anomaly_score(flatten_scaled_valid, flatten_pred_valid)

        # thresholding
        # thrsehold = anoLSTM().find_criteria(true, pred, labels)

        thrsehold_otsu = thr.otsu_threshold(err_values)
        thrsehold_iqr = thr.iqr_threshold(ano_score, p_high=95, p_low=5)
        print(thrsehold_otsu, thrsehold_iqr)

        err_cnt = err_values[np.where(err_values > thrsehold_otsu)]
        ano_cnt = np.concatenate((ano_score[np.where(thrsehold_iqr[1] > ano_score)], ano_score[np.where(ano_score > thrsehold_iqr[0])]))
        print(err_cnt.size, ano_cnt.size)

        # chart
        thr_otsu = [thrsehold_otsu for i in range(err_values.size)]
        thr_iqr_high = [thrsehold_iqr[0] for i in range(ano_score.size)]
        thr_iqr_low = [thrsehold_iqr[1] for i in range(ano_score.size)]

        plt.plot(ano_score)
        plt.plot(thr_iqr_high, c='red')
        plt.plot(thr_iqr_low, c='red')

        plt.show()
