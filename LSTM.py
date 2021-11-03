import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
timesteps = 41
def get_data():
    close_data = pd.read_csv('C:/Users/83862/Desktop/LSTM/Nasdaq100.csv')
    ed_data=close_data.iloc[:,1:]
    return ed_data

def data_pre(data):
    sc = MinMaxScaler(feature_range=(0, 1))
    sc_data = sc.fit_transform(data)
    sc_data = np.nan_to_num(sc_data)
    X_train = []
    y_train = []
    for i in range(timesteps, data.shape[0] - timesteps - 16):
        X_train.append(sc_data[i - timesteps:i, :])
        y_train.append(sc_data[i, :])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    inputs = sc_data[-16 - timesteps:-16]
    X_test = []
    X_test.append(inputs[0:timesteps])
    X_test = np.array(X_test)

    X_val = []
    y_val = []
    for i in range(data.shape[0] - timesteps - 16, data.shape[0] - 16):
        X_val.append(sc_data[i - timesteps:i, :])
        y_val.append(sc_data[i, :])
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, y_train, X_test, X_val, y_val, sc

def predict(X_train, y_train, close_data, X_test, X_val, y_val, sc):
    regressor = Sequential()
    regressor.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(LSTM(units=200))
    regressor.add(Dense(102))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    history = regressor.fit(X_train, y_train, epochs=15, batch_size=10, verbose=2)
    plt.plot(history.history['loss'], 'r-', alpha=0.4)
    plt.title('training loss')
    plt.show()

    y_pred = regressor.predict(X_val, verbose=0)
    rmse = np.sqrt(np.mean(np.power((sc.inverse_transform(y_val) - sc.inverse_transform(y_pred)), 2)))

    predictions = []
    for j in range(timesteps, timesteps + 16):
        predict = regressor.predict(X_test[0, j - timesteps:j].reshape(1, timesteps, 102))
        predict1 = sc.inverse_transform(predict)
        X_test = np.append(X_test, predict).reshape(1, j + 1, 102)
        predictions.append(predict1)
    final = close_data.iloc[0:16, 0:]
    for i in range(0, 16):
        final.iloc[i, :] = predictions[i]
    return final, rmse

def main():
    X_train,y_train,X_test,X_val,y_val,sc = data_pre(get_data())
    final,rmse = predict(X_train,y_train,get_data(),X_test,X_val,y_val,sc)
    return final,rmse

final,rmse = main()
