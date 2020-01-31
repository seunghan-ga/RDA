from RDA.analysis.anoLSTM import anoLSTM
import tensorflow as tf
import pandas as pd
import pymysql
import joblib


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
    cursor.execute("select * from sar limit 1000")
    columns = ["date", "type", "user", "nice", "system", "iowait", "steal", "idel"]
    data = pd.DataFrame(cursor.fetchall(), columns=columns)

    scaler = joblib.load("RDA/models/normalization.scaler")
    json_file = open("RDA/models/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("RDA/models/model.h5")
    LSTM_Model = loaded_model

    with tf.device('/GPU:0'):
        test, _ = anoLSTM().data_preparation(data[columns[2:]].astype("float").values, scaler, 3, 3)
        true, pred = anoLSTM().prediction(LSTM_Model, scaler, test)

        ano_score = anoLSTM().anomaly_score(true, pred)
        labels = anoLSTM().labeling(true, 6)
        thrsehold = anoLSTM().find_criteria(true, pred, labels)
        f1_test, pred_test, ab_idx = anoLSTM().f1score(ano_score, labels, thrsehold)
        import matplotlib.pyplot as plt
        import numpy as np
        print(ano_score)
        print(np.mean(ano_score))
        print(len(ano_score[ano_score < np.mean(ano_score)]))
        plt.plot(ano_score[ano_score < np.mean(ano_score)])
        plt.show()

