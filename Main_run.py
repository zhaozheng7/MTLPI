import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from model import PIModel
from sklearn.metrics import r2_score
from Earlystop import EarlyStop

tf.keras.utils.set_random_seed(2)
tf.config.experimental.enable_op_determinism()

# TODO: 1. Add the loss function for the model; this is the main part of the paper.
# TODO: 2. Add CAGrad class to the main program.
"""
1.QD+无法收敛问题解决
2.加入CAGrad
"""

class Run:
    def __init__(self, normal):
        self.normal = normal
        self.normal_model = normal.model
        # To keep the results
        self.result = []
        self.opt = tf.keras.optimizers.legacy.Adam()
        """
        这里设置损失函数名称，目前支持QD, QD+, Continuous, Dual, MTLPI
        """
        self.loss_name = 'QD+'

    @classmethod
    def set_epochs(cls, epochs):
        cls.epochs = epochs

    @classmethod
    def set_batch_size(cls, batch_size):
        cls.batch_size = batch_size

    def run_normal(self, file_path):
        self.normal_model.init_arguments(loss_name=self.loss_name, coverage_rate=0.99, method='CAGrad')
        self.normal_model.compile(optimizer=self.opt,
                                  loss=[self.normal_model.selective_up,
                                        self.normal_model.selective_low,
                                        self.normal_model.up_penalty,
                                        self.normal_model.low_penalty,
                                        self.normal_model.coverage_penalty,
                                        self.normal_model.abs_selective_up,
                                        self.normal_model.abs_selective_low, ],
                                  metrics=[self.normal_model.coverage, self.normal_model.mpiw,
                                           self.normal_model.mse_penalty])

        def lr_scheduler(epoch):
            learning_rate = 0.001
            lr_drop = 1000
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        history_normal = self.normal_model.fit(self.normal.X_train, self.normal.y_train,
                                               validation_data=(self.normal.X_val, [self.normal.y_val[:, 0],
                                                                                    self.normal.y_val[:, 1]]),
                                               batch_size=self.batch_size,
                                               epochs=self.epochs,
                                               callbacks=[reduce_lr, EarlyStop(patience=600)],
                                               verbose=1)
        # # Save the training history for analysis
        name = self.normal.dataset.split('.')[0]

        with open(os.path.join(file_path, f'{name}_history.pkl'), 'wb') as handle:
            pickle.dump(history_normal.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # To plot and save .png
        self.plot_training(os.path.join(file_path, f'{name}_history.pkl'), file_path)
        # Predicted results
        normal_pred = self.normal_model.predict(self.normal.X_test)
        return normal_pred

    def save_pred_results(self, model, predicitons, name, file_path):
        df = pd.DataFrame(predicitons, columns=['Lowerbound', 'Upperbound', 'PredPoint'])
        # df['y_true'] = model.reversed_norm(model.y_test[:,0].reshape(-1,1))
        df['y_true'] = model.y_test[:, 0]
        df['Width'] = (df['Upperbound'] - df['Lowerbound'])
        df['MPIW'] = np.mean(df['Width'])

        df['Flag'] = np.where((df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0)
        df['PICP'] = np.mean((df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']))
        # To sort the columns
        df = df[['PICP', 'Lowerbound', 'y_true', 'PredPoint', 'Upperbound', 'Flag', 'MPIW', 'Width']]  # 'NMPIW',
        # Save all predicted value
        dataset_name = model.dataset.split('.')[0]
        df.to_csv(os.path.join(file_path, f'{dataset_name}_{name}_pred.csv'), header=True, index=False)
        self.result.append({'PICP': np.mean(df['PICP']), 'MPIW': np.mean(df['MPIW'])})  # ,'NMPIW':np.mean(df['NMPIW'])

    def print_res(self):
        res = pd.DataFrame(self.result)
        print(res)
        return res

    def plot_prediction_intervals(self, model, predictions, file_path, name, loss, cver):
        # To plot the prediction intervals
        r2 = r2_score(model.y_test[:, 2], predictions[:, 2])
        rmse = np.sqrt(np.mean((model.y_test[:, 2] - predictions[:, 2]) ** 2))
        print(f"R2:   {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        plt.figure(figsize=(10, 6))
        plt.plot(model.X_all, model.y_all, 'o', markersize=5, label='TrueData')
        plt.plot(model.X_test, predictions[:, 0], 'o', markersize=5.5, label='Lowerbound')
        plt.plot(model.X_test, predictions[:, 1], 'o', markersize=5.5, label='Upperbound')
        plt.plot(model.X_test, predictions[:, 2], 'o', markersize=5.5, label='PredPoint')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{name}_{cver}_{loss}_{r2:.4f}_{rmse:.4f}')
        plt.legend()
        plt.savefig(os.path.join(file_path, f'{name}_{cver}_{loss}.png'), dpi=300)

    def plot_training(self, filename, file_path):
        name = filename.split('.')[0]
        dict_data = pd.read_pickle(filename)
        df = pd.DataFrame(dict_data)
        plt.figure(figsize=(10, 6))
        plt.ylim(0, 5)
        sns.set_style("ticks")
        plt.title(name)
        plt.xlabel("Epochs")
        sns.lineplot(
            data=df[['coverage', 'mpiw', 'mse_penalty', 'val_coverage', 'val_mpiw', 'val_loss', 'val_mse_penalty']])
        plt.savefig(os.path.join(file_path, f'{name}.png'), dpi=300)
        # plt.clf()
        # plt.show()

    def save_aggregation_pred_results(self, model, predicitons, name, file_path):
        """
        reload the function to save aggregation results
        :param model:
        :param predicitons:
        :param name:
        :return:
        """
        df = pd.DataFrame(predicitons, columns=['Lowerbound', 'Upperbound', 'PredPoint'])
        # df['y_true'] = model.reversed_norm(model.y_test[:,0].reshape(-1,1))
        df['y_true'] = model.y_test[:, 0]
        df['Width'] = (df['Upperbound'] - df['Lowerbound'])
        df['MPIW'] = np.mean(df['Width'])

        df['Flag'] = np.where((df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0)
        df['PICP'] = np.mean((df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']))
        # To sort the columns
        df = df[['PICP', 'Lowerbound', 'y_true', 'PredPoint', 'Upperbound', 'Flag', 'MPIW', 'Width']]  # 'NMPIW',
        # Save all predicted value
        dataset_name = model.dataset.split('.')[0]
        df.to_csv(os.path.join(file_path, f'{dataset_name}_{name}_pred_aggregation.csv'), header=True, index=False)
        return np.mean(df['PICP']), np.mean(df['MPIW'])


def main():
    """
    using paper data 重要，这里设置数据集和目标值
    """
    dataset = ['101_datausingIEEE.csv']
    target = ['y']
    # dataset = ['concrete_data.csv']
    # target = ['concrete_compressive_strength']
    # dataset = ['boston_housing_data.csv']
    # target = ['MDEV']
    """ energy setting: """
    # dataset = ['energy_data.csv']
    # target = ['Y1']
    # dataset = ['kin8nm_data.csv']
    # target = ['y']
    # dataset = ['naval_data.csv']
    # target = ['GTTurb']
    # dataset = ['power_data.csv']
    # target = ['PE']
    # dataset = ['protein_data.csv']
    # target = ['RMSD']
    # dataset = ['winequality-red.csv']
    # target = ['quality']
    # dataset = ['yacht_data.csv']
    # target = ['Y']
    if dataset == ['101_datausingIEEE.csv']:
        is_plot = True
    else:
        is_plot = False
    name = dataset[0].split('.')[0]
    file_path = "E:\\VIcurve\\MTL_PredictionIntervals\\result\\" + name + "\\"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    def training_data(datasets, targets):
        """
        嵌套在main程序中，用于迭代训练数据
        :param datasets:
        :param targets:
        :return:none
        """
        global obj
        for index, (dataset, target) in enumerate(zip(datasets, targets)):
            if dataset == 'protein_data.csv':
                times = 5
            elif dataset == 'yearMSD_data.csv':
                times = 1
            else:
                times = 1
            temp = []
            result_list = []
            name = dataset.split('.')[0]

            # 运行次数为times
            for i in range(times):
                # 种子可随机复现
                # seed = np.random.randint(100)
                seed = 22
                # 设置实例，其中dataset和target作为一次训练数据输入
                normal = PIModel(dataset, target, seed=seed)
                # To calculate the data size
                if normal.y_test.shape[0] < 40.0:
                    normal = PIModel(dataset, target, drop_out=0.25, flag=True, seed=seed)
                    # initial_weights = normal.model.get_weights()
                else:
                    initial_weights = normal.model.get_weights()
                obj = Run(normal)

                normal_Pred = obj.run_normal(file_path)
                obj.save_pred_results(normal, normal_Pred, 'normal_pred', file_path)
                cver = obj.normal_model.coverage_rate
                loss = obj.normal_model.loss_name
                if is_plot:
                    obj.plot_prediction_intervals(normal, normal_Pred, file_path, name, loss, cver)
                res = obj.print_res()
                temp.append(res)
                result_list.append(normal_Pred)

            output = pd.concat(temp)
            # name = dataset.split('.')[0]
            cver = obj.normal_model.coverage_rate
            loss = obj.normal_model.loss_name

            # output.to_csv(f'{name}_{cver}_{loss}_Outputs.csv')
            output.to_csv(os.path.join(file_path, f'{name}_{cver}_{loss}_Outputs.csv'))

    training_data(dataset, target)


if __name__ == "__main__":
    # 设置迭代次数
    Run.epochs = 3000
    # 设置批大小
    Run.batch_size = 256
    # 运行主程序
    main()
    # 图片展示
    plt.show()
