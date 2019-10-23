
import pandas as pd
import numpy as np
from datetime import datetime


# 1.读取数据，并获取数据缺失值
def process_data(file_name):
    df = pd.read_table(file_name, header=None)
    # df=pd.read_table("F:/weibo_train_data.txt",header=None)
    df.columns = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
    df_1 = df.dropna(axis=0)  # 删除缺失值所在行
    print("the null rows is: ", len(df) - len(df_1))

    return df_1

# 2.观察数据整体分布情况

# 单独获取每条博文的情况
def get_binary(data, column_name):
    """
    分别查看博文的点赞数、评论数和转发数
    :param data: 数据框
    :param column_name:
    :return:
    """
    data_column_1 = sum(data[column_name] > 0)
    total_num = len(data)
    residual = total_num - data_column_1
    return data_column_1, residual

# 获取每条博文转发,评论和点赞的数量情况,分为8类
def get_num(data, column_list):
    """
    共有八种类别（点赞数，转发数，评论数）：
        （0,0,0）,（1,0,0）,（0,1,0）, （0,0,1）
         (1,1,0),  (1,0,1),  (1,1,0),  (1,1,1)
    :param data:
    :param column_list: 因变量所在列名
    :return:
    """
    label = pd.DataFrame(np.zeros((len(data), 3)))
    label.columns = column_list
    for i in range(len(data)):
        for column in column_list:
            try:
                if data.loc[i, column] > 0:
                    label.loc[i, column] = 1
            except:
                continue
    return label
