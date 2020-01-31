from RDA.analysis.anoLSTM import anoLSTM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import pandas as pd
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
    cursor.execute("select * from sar")
    columns = ["date", "type", "user", "nice", "system", "iowait", "steal", "idel"]
    data = pd.DataFrame(cursor.fetchall(), columns=columns)
    train, scaler = anoLSTM().data_preparation(data[columns[2:]].astype("float").values, None, n_in=3, n_out=3)

    joblib.dump(scaler, "RDA/models/normalization.scaler")
    print("Saved scaler to disk")

    with tf.device('/GPU:0'):
        LSTM_Model = anoLSTM().lstm_autoencoder(train, n_in=3, hidden_dim=50)
        LSTM_Model.summary()
        history = LSTM_Model.fit(train, train, epochs=100, verbose=2, batch_size=64,
                                 shuffle=True, validation_split=0.2)

        model_json = LSTM_Model.to_json()  # serialize model to JSON
        json_file = open("RDA/models/model.json", "w")
        json_file.write(model_json)
        LSTM_Model.save_weights("RDA/models/model.h5")  # serialize weights to HDF5
        print("Saved model to disk")

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')

        plt.show()
