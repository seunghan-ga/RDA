from RDA.analysis.anoLSTM import anoLSTM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
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
    data = data[columns[2:]].astype("float").values

    timesteps = 30  # window size
    features = columns[1:]
    n_features = 6
    units = len(features) * 2
    epochs = 1000
    batch_size = 128

    train = anoLSTM().preparation(X=data, timesteps=timesteps, n_features=n_features)
    flat_train = anoLSTM().flatten(train)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(anoLSTM().flatten(train))
    scaled_train = anoLSTM().scaling(train, scaler)

    with tf.device('/GPU:0'):
        LSTM_Model = anoLSTM().get_model(timesteps=timesteps,
                                         n_features=n_features,
                                         units=units,
                                         activate_func=tf.keras.activations.tanh,
                                         recurrent_func=tf.keras.activations.sigmoid,
                                         kernel_init=tf.keras.initializers.glorot_uniform(),
                                         loss_func=tf.keras.losses.mean_squared_error,
                                         optimize_func=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                                         dropout=0.3)
        LSTM_Model.summary()

        MODEL_SAVE_PATH = "RDA/models/checkpoints/"
        model_path = MODEL_SAVE_PATH + '{epoch:03d}-{val_loss:.5f}.hdf5'
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1,
                                                           save_best_only=True)
        cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        history = LSTM_Model.fit(x=scaled_train,
                                 y=scaled_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=2,
                                 callbacks=[cb_checkpoint, cb_early_stopping],
                                 validation_split=0.2,
                                 shuffle=False).history

        joblib.dump(scaler, "RDA/models/normalization.scaler")
        print("Saved scaler to disk")

        model_json = LSTM_Model.to_json()  # serialize model to JSON
        json_file = open("RDA/models/model.json", "w")
        json_file.write(model_json)
        LSTM_Model.save_weights("RDA/models/model.h5")  # serialize weights to HDF5
        print("Saved model to disk")

        plt.plot(history['loss'], label='train')
        plt.plot(history['val_loss'], label='test')

        plt.show()
