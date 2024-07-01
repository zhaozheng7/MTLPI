import numpy as np
import pandas as pd

if __name__ == '__main__':

    filepath = 'result/power_data/power_data_0.95_MTLPI_Outputs.csv'
    df = pd.read_csv(filepath)
    print(df)

    # 计算平均值
    PICP_mean = df['PICP'].mean()
    # 计算标准差
    PICP_std = df['PICP'].std()
    # 计算平均值
    MPIW_mean = df['MPIW'].mean()
    # 计算标准差
    MPIW_std = df['MPIW'].std()
    # 计算平均值
    MSE_mean = df['MSE'].mean()
    # 计算标准差
    MSE_std = df['MSE'].std()
    print("PICP: ","{:.3f}".format(PICP_mean), "+-", "{:.3f}".format(PICP_std))
    print("MPIW:","{:.3f}".format(MPIW_mean), "+-", "{:.3f}".format(MPIW_std))
    print("MSE:", "{:.3f}".format(MSE_mean), "+-", "{:.3f}".format(MSE_std))