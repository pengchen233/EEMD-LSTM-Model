import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pandas import concat


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from numpy import concatenate

from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy import interpolate, math
import matplotlib.pyplot as plt
import matplotlib as mpl

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model


plt.rcParams['figure.figsize'] = (10.0, 5.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

dataset = pd.read_csv('../csv/Water Quality Record.csv', header=0, index_col=0, parse_dates = True)
data = dataset.values.reshape(-1)

values = dataset.values
groups = [ 0, 1, 2, 3]
fig, axs = plt.subplots(1)


# i = 1
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])
#     plt.title(dataset.columns[group], fontsize=30)
#     i+=1

#展示溶解氧那一列
# plt.subplot(1, 1, 1)
# plt.plot(values[:, 3])
# plt.title(dataset.columns[3], fontsize=30)

# plt.savefig('/Users/core/Desktop/fig3.png')
# plt.show()

df=pd.DataFrame(dataset)  #整体数据的全部字典类型
do=df['Dissolved Oxygen']  #返回溶解氧那一列，用字典的方式

DO=[]
for i in range(0,len(do)):
    DO.append([do[i]])
scaler_DO = MinMaxScaler(feature_range=(0,1))
DO = scaler_DO.fit_transform(DO)
plt.plot(DO)


# plt.show()
#DO前后类型发生变化，list变为numpy.ndarray
#DO(3000, 1)

#lookback_window ：回望窗口
#用多少值预测一个值
def data_split(data, train_len, lookback_window):
    train = data[:train_len]  #标志训练集
    test = data[train_len:]   #标志测试集
    # print(train.shape)

    #X1[]代表移动窗口中的10个数
    #Y1[]代表相应的移动窗口需要预测的数
    #X2, Y2 同理

    X1, Y1 = [], []
    for i in range(lookback_window, len(train)):
        X1.append(train[i - lookback_window:i])
        Y1.append(train[i])
    Y_train = np.array(Y1)
    X_train = np.array(X1)

    X2, Y2 = [], []
    for i in range(lookback_window, len(test)):
        X2.append(test[i - lookback_window:i])
        Y2.append(test[i])
    Y_test = np.array(Y2)
    X_test = np.array(X2)

    print(X_train.shape)
    print(Y_train.shape)
    return (X_train, Y_train, X_test, Y_test)

#shape是查看数据有多少行多少列
#reshape()是数组array中的方法，作用是将数据重新组织

def data_split_LSTM(X_train,Y_train, X_test, Y_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)
    return (X_train, Y_train, X_test, Y_test)

#
def imf_data(data, lookback_window):
    # train = data[:train_len]  # 标志训练集
    # test = data[train_len:]  # 标志测试集
    # print(train.shape)

    # X1[]代表移动窗口中的10个数
    # Y1[]代表相应的移动窗口需要预测的数
    # X2, Y2 同理

    X1 = []
    for i in range(lookback_window, len(data)):
        X1.append(data[i - lookback_window:i])
    X1.append(data[len(data)-1:len(data)])
    X_train = np.array(X1)

    return (X_train)



def visualize(history):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def LSTM_Model(X_train, Y_train):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))   #已经确定10步长
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=100, batch_size=1, validation_split=0.1, verbose=1, shuffle=True)
    return (model)
# batch_size=1 有最优的结果, 但是需要更长的时间

c=int(len(df)*.9)

#数组划分为不同的数据集
X1_train, Y1_train, X1_test, Y1_test =data_split(DO, c, 10) #TCN
X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
# print(X2_train.ndim)     #3
# print(X2_test.ndim)      #3
# print(Y2_train.ndim)     #2
# print(Y2_test.ndim)      #2
# print(X2_train.shape)       #(2690, 10, 1)

#有趣的测试
# print(X2_train[0][0])     #3
# print(X2_test[0][0])      #3
# print(Y2_train[0][0])     #2
# print(Y2_test.ndim)      #2

#训练模型
# model_DO_LSTM=LSTM_Model(X2_train, Y2_train)

#记载已经保存的模型
model_DO_LSTM=load_model('res/lstm_model_100.h5')

#保存模型
# model_DO_LSTM.save('lstm_model_100.h5')

#画出曲线变化图
def plot_curve(true_data, predicted):
    plt.plot(true_data, label='True data')
    plt.plot(predicted, label='Predicted data')
    # plt.plot(predicted_LSTM, label='Predicted data by LSTM') plt.legend()
    plt.savefig('result_final.png')
    plt.show()

#定义RMSE、MAPE两个指标
def RMSE(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse

def MAPE(Y_true, Y_pred):
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    return np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100

Y2_train_hat=model_DO_LSTM.predict(X2_train)
#变回原来的值,inverse_transform
Y2_train_hat=scaler_DO.inverse_transform(Y2_train_hat)
Y2_train=scaler_DO.inverse_transform(Y2_train)
# print(Y2_train.ndim)
# print(Y2_train_hat.ndim)都是2维数组

Y2_test_hat=model_DO_LSTM.predict(X2_test)
Y2_test_hat=scaler_DO.inverse_transform(Y2_test_hat)
Y2_test=scaler_DO.inverse_transform(Y2_test)

plot_curve(Y2_train, Y2_train_hat)
plot_curve(Y2_test, Y2_test_hat)

# print(RMSE(Y2_test,Y2_test_hat))
# print(MAPE(Y2_test,Y2_test_hat))
