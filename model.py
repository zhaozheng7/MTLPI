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


class PIModel:

    def __init__(self, dataset, target, drop_out=0.25, flag=False, seed=2):

        self.filepath = 'E:\VIcurve\MTL_PredictionIntervals\dataset\\'
        self.dataset = dataset
        self.target = target

        # To ensure the PCG and No_PCG use the same training and testing data each time
        self.seed = seed
        self.data_pre()
        self.drop_out = drop_out

        # To control the drop_out for 'big' and small datasize
        self.flag = flag
        self.model = self.build_model()

    # Custom a training model
    def build_model(self):

        inputs = Input(shape=self.X_train.shape[1:])

        """
        20240612 add by zz
        添加预测值，用于预测和模型特征扩充。
        """
        # 预测点模型
        x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu')(inputs)
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
        lower_bound = layers.Dense(1, bias_initializer=keras.initializers.constant(-4.0),  # 4
                                   name='lower_bound')(lower_bound)

        # Upper bound  (head)
        upper_bound = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='linear')(curr)
        if self.flag:
            upper_bound = layers.Dropout(self.drop_out)(upper_bound)

        upper_bound = layers.Dense(1, bias_initializer=keras.initializers.constant(4.0),
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
        self.X_all, self.y_all = self.standard_scaler(X, y)
        # For comparison 1: training, validation, test data (0.7, 0.2, 0.1)
        # For comparison 2: training, validation, test data (0.81, 0.09, 0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=self.seed)

        X_train = X_train.astype('float32')
        y_train = y_train.astype('float32')

        X_val = X_val.astype('float32')
        y_val = y_val.astype('float32')

        X_test = X_test.astype('float32')
        y_test = y_test.astype('float32')

        # For comparison 1
        # self.X_train, self.y_train = self.minmax_scaler(X_train, y_train)
        # For comparison 2
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
