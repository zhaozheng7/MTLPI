import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras import regularizers
from tensorflow.keras import layers
from Trainstep import PI_Controler
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm


class PIModel:

    def __init__(self, dataset, target, drop_out=0.25, flag=False, seed=2):

        self.filepath = 'E:\VIcurve\MTL_PredictionIntervals\dataset\\'
        self.dataset = dataset
        self.target = target

        # To ensure the PCG and No_PCG use the same training and testing data each time
        self.seed = seed
        if self.dataset.startswith('VI'):
            self.pre_vi_data()
        else:
            self.data_pre()
        self.drop_out = drop_out

        # To control the drop_out for 'big' and small datasize
        self.flag = flag
        self.model = self.build_model()
        self.method = 'normal'

    # Custom a training model
    def build_model(self):

        inputs = Input(shape=self.X_train.shape[1:])

        """
        20240612 add by zz
        添加预测值，用于预测和模型特征扩充。
        """
        # 预测点模型
        x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001), activation='relu')(inputs)
        x = layers.Dense(64, activation='linear')(x)
        pred_point = layers.Dense(1, name='pred_point')(x)
        # 将预测点作为新特征
        ddd = layers.Concatenate()([inputs, pred_point])

        # curr means shared bottom
        curr = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                            kernel_initializer='normal')(ddd)
        curr = layers.Dropout(self.drop_out)(curr)

        curr = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                            kernel_initializer='normal')(curr)
        curr = layers.Dropout(self.drop_out)(curr)

        # Lower bound  (head)
        lower_bound = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='linear')(curr)
        if self.flag:
            lower_bound = layers.Dropout(self.drop_out)(lower_bound)
        lower_bound = layers.Dense(1, bias_initializer=keras.initializers.constant(-1.0),  # 4
                                   name='lower_bound')(lower_bound)

        # Upper bound  (head)
        upper_bound = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='linear')(curr)
        if self.flag:
            upper_bound = layers.Dropout(self.drop_out)(upper_bound)

        upper_bound = layers.Dense(1, bias_initializer=keras.initializers.constant(1.0),
                                   name="upper_bound")(upper_bound)

        """20240529 changed by zz"""
        """add one layer to calculate points prediction"""
        # mid point  (head)
        # mid_point = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='linear')(curr)
        # if self.flag:
        #     mid_point = layers.Dropout(self.drop_out)(mid_point)
        # mid_point = layers.Dense(1, name='mid_point')(mid_point)
        # 输出
        """add one output to get the point prediction"""
        outputs = Concatenate(axis=1, name="combined_output")([lower_bound, upper_bound, pred_point])  # 下界，上界，预测值

        return PI_Controler(inputs, outputs)

    def data_pre(self):
        file_path = os.path.join(self.filepath, self.dataset)
        if file_path.split('.')[-1] == 'xls' or file_path.split('.')[-1] == 'xlsx':
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        if self.dataset == 'energy_data.csv':
            df.pop('Y2')
            y = df.pop(self.target)
            y = y.to_numpy().reshape(-1, 1)
            y = y[0:768]
            X = df.to_numpy()
            X = X[0:768, 0:8]
        else:
            y = df.pop(self.target)
            y = y.to_numpy().reshape(-1, 1)
            X = df.to_numpy()

        # Save the range for comparison 1
        # self.range = np.max(y) - np.min(y)

        if self.dataset == 'syntheticdata.csv':
            """
            train:0.8
            test:
            """
            self.X_all, self.y_all = X, y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=self.seed)
        else:
            self.X_all, self.y_all = self.standard_scaler(X, y)
            # train : 0.8 val: 0.1 test: 0.1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=self.seed)

        X_train = X_train.astype('float32')
        y_train = y_train.astype('float32')

        X_val = X_val.astype('float32')
        y_val = y_val.astype('float32')

        X_test = X_test.astype('float32')
        y_test = y_test.astype('float32')

        # For comparison 1
        # self.X_train, self.y_train = self.minmax_scaler(X_train, y_train)
        # For comparison 2
        if self.dataset == 'syntheticdata.csv':
            self.X_train, self.y_train = X_train, y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
        else:
            self.X_train, self.y_train = self.standard_scaler(X_train, y_train)

            self.X_val = self.scaler_x.transform(X_val)
            self.y_val = self.scaler_y.transform(y_val)

            self.X_test = self.scaler_x.transform(X_test)
            self.y_test = self.scaler_y.transform(y_test)

        # Change y shape based on the model
        self.y_train = np.repeat(self.y_train, [3], axis=1)
        self.y_val = np.repeat(self.y_val, [3], axis=1)
        self.y_test = np.repeat(self.y_test, [3], axis=1)

    # For comparison 1
    def minmax_scaler(self, X, y):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        X = self.scaler_x.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        return X, y

        # For comparison 2

    def standard_scaler(self, X, y):
        self.scaler_x = StandardScaler()

        self.scaler_y = StandardScaler()
        X = self.scaler_x.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        return X, y

        # Invert transform on predictions

    def reversed_norm(self, yhat):
        return self.scaler_y.inverse_transform(yhat)

    # Z-score 标准化函数
    def z_score_normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / std

    def pre_vi_data(self, vi_split=False, plot=False):
        if self.dataset.startswith('VI_') and vi_split == False:
            '''
            判断当前使用的数据集为VI模式 格式例如VI_2-1
            '''
            # 提取文件名中数字部分
            Normal_flag = True  # 是否归一化
            dataset_name = self.dataset.split('_')[1]
            # 初始路径
            # path_ = "E:\\VIcurve\\Project\\PredictionIntervals-master\\Normal"
            path_ = self.filepath + '\\' + 'Normal_nocircle'
            # 根据提取的数字部分加入到初始路径中
            path__ = path_ + '\\' + dataset_name + 'normal'
            # 进入这一层目录，获得该文件夹中所有txt文件
            file_list = os.listdir(path__)
            xfortrain = []
            yfortrain = []
            down_all_final = np.empty(shape=[0, 2])
            up_all_final = np.empty(shape=[0, 2])
            for i in tqdm(range(len(file_list))):  # 这里获得新的路径
                ppaatthh = path__ + '/' + file_list[i]
                # print("正在处理该路径文件" + ppaatthh)
                # 求单值函数训练所需的x和y
                x_all, y_all = self.pd_ReadData(ppaatthh, Normal_flag=Normal_flag)
                xfortrain.extend(x_all)
                yfortrain.extend(y_all)
            xfortrain = np.array(xfortrain)
            yfortrain = np.array(yfortrain)
            print("数据：" + path__ + " 加载完成")
            # 分为x和y数据
            X = xfortrain.flatten()
            X = np.reshape(X, (len(X), 1))
            Y = yfortrain.flatten()
            print(X.shape, Y.shape)
            if plot == True:
                plt.figure()
                plt.scatter(X, Y, s=0.5)
                plt.show()
            # return X, Y
            # usefortrain = np.array([X,Y]).T
            # usefortrain = pd.DataFrame(usefortrain, columns=['V', 'I'])
            self.X_all, self.y_all = X, Y
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=self.seed)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=self.seed)
            X_train = X_train.astype('float32').reshape(-1, 1)
            y_train = y_train.astype('float32').reshape(-1, 1)

            X_val = X_val.astype('float32').reshape(-1, 1)
            y_val = y_val.astype('float32').reshape(-1, 1)

            X_test = X_test.astype('float32').reshape(-1, 1)
            y_test = y_test.astype('float32').reshape(-1, 1)
            self.X_train, self.y_train = X_train, y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
            # Change y shape based on the model
            self.y_train = np.repeat(self.y_train, [3], axis=1)
            self.y_val = np.repeat(self.y_val, [3], axis=1)
            self.y_test = np.repeat(self.y_test, [3], axis=1)
        elif (self.dataset.startswith('VI_') and vi_split == True):
            '''
            判断当前使用的数据集为VI模式 格式例如VI_2-1
            '''
            # 提取文件名中数字部分
            Normal_flag = True
            dataset_name = self.dataset.split('_')[1]
            # 初始路径
            # path_ = "E:\\VIcurve\\Project\\PredictionIntervals-master\\Normal"
            path_ = self.filepath + '\\' + 'Normal_circle'
            # 根据提取的数字部分加入到初始路径中
            path__ = path_ + '\\' + dataset_name + 'normal'
            # 进入这一层目录，获得该文件夹中所有txt文件
            file_list = os.listdir(path__)
            xfortrain = []
            yfortrain = []
            down_all_final = np.empty(shape=[0, 2])
            up_all_final = np.empty(shape=[0, 2])
            for i in tqdm(range(len(file_list))):  # 这里获得新的路径
                ppaatthh = path__ + '/' + file_list[i]
                # print("正在处理该路径文件" + ppaatthh)
                # 多值函数训练所需的电压上升段和下降段数据
                downgrp, upgrp = self.datacut(ppaatthh, Normal_flag=Normal_flag)  # 分成上下段
                down_all_final = np.vstack((down_all_final, downgrp))
                up_all_final = np.vstack((up_all_final, upgrp))
            print("数据：" + path__ + " 加载完成")
            if plot == True:
                plt.figure()
                plt.scatter(down_all_final[:, 0], down_all_final[:, 1], c='r', label='down', s=0.5)
                plt.legend()
                plt.figure()
                plt.scatter(up_all_final[:, 0], up_all_final[:, 1], c='b', label='up', s=0.5)
                plt.legend()
                plt.show()
            print(down_all_final.shape, up_all_final.shape)
            return down_all_final, up_all_final

    def pd_ReadData(self, path, Normal_flag=True, **kwargs):
        """
           使用pandas库来获取数据
           :param path:
           :Normal_flag: 归一化标志
           :param kwargs:
           :return: 数据x和数据y
           """
        data = pd.read_table(path, sep='\s+', encoding='UTF-8', skiprows=3, usecols=(0, 1), header=None)
        data = np.array(data)
        x = data[:, 1].reshape(-1, 1)
        y = data[:, 0].reshape(-1, 1)
        if Normal_flag:
            scaler = preprocessing.MinMaxScaler()
            x = scaler.fit_transform(x)
            y = scaler.fit_transform(y)
        return x, y

    def datacut(self, path, Normal_flag=True):
        data_all = []
        x_mm, y_mm = self.pd_ReadData(path, Normal_flag)  # txt文件归一化处理
        data_all = np.hstack((x_mm, y_mm))

        upgrp = []
        downgrp = []
        data_all = np.array(data_all)
        i = len(data_all)

        # 获得单个TXT文件中的最大最小值以及索引
        index_max = np.argmax(data_all[:, 0])
        max_val = np.max(data_all[:, 0])
        index_min = np.argmin(data_all[:, 0])
        min_val = np.min(data_all[:, 0])
        if index_max > index_min:  # up
            d_temp1 = data_all[index_min:index_max, :].reshape(-1, 2)
            # a_temp1 = data_all[index_min:index_max,0].reshape(1,-1)
            d_tempd = data_all[0:index_min, :].reshape(-1, 2)
            d_tempd_ano = data_all[index_max:100, :].reshape(-1, 2)
            downgrp = np.concatenate((d_tempd, d_tempd_ano), axis=0)
            upgrp = d_temp1

        else:
            d_temp2 = data_all[index_max:index_min, :].reshape(-1, 2)
            d_tempd = data_all[0:index_max, :].reshape(-1, 2)
            d_tempd_ano = data_all[index_min:100, :].reshape(-1, 2)
            downgrp = d_temp2
            upgrp = np.concatenate((d_tempd, d_tempd_ano), axis=0)

        return downgrp, upgrp
