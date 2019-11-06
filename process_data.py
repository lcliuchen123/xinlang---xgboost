
import pandas as pd
import numpy as np

"""# 注意：read_table读取数据会出现丢失，原因是某行中有单个英文引号，导致\n换行符失效。
# 多行连成了一行，直到遇到下一个单个引号。
参考：https://blog.csdn.net/weixin_42294274/article/details/89518338
# 解决方法:设置quoting=3
# 参数：quoting : int or csv.QUOTE_* instance, default 0
# 控制csv中的引号常量。可选 QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or QUOTE_NONE (3)
# 参数：doublequote : boolean, default True
# 双引号，当单引号已经被定义，并且quoting 参数不是QUOTE_NONE的时候，
# 使用双引号表示引号内的元素作为一个元素使用。"""

# 1.读取数据，并获取数据缺失值
def process_data(file_name,type):
    try:
        df = pd.read_table(file_name, header=None,encoding='utf-8',quoting=3)
        print("the total rows is: ", len(df))
        print("the null rows is: ", sum(df.isnull().any(axis=1)))
        # df=pd.read_table("F:/weibo_train_data.txt",header=None)
        if type == 'train':
            df.columns = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
            df_1 = df.dropna(axis=0)  # 删除缺失值所在行
            return df_1
        elif type == 'test':
            print('test')
            df.columns = ['uid', 'mid', 'time',  'content']
            df_1 = df.fillna(method='ffill') #前向填充
            return df_1
        else:
            print("ERROR!!! Please input the correct type: train or test")
    except Exception as e:
        print(e)

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

# 3.按照每条博文转发,评论和点赞的数量情况,分为8类
def get_num(data, column_list):
    """
    共有八种类别（点赞数，转发数，评论数）：
        （0,0,0）,（1,0,0）,（0,1,0）, （0,0,1）
         (1,1,0),  (1,0,1),  (1,1,0),  (1,1,1)
    :param data:
    :param column_list: 因变量所在列名
    :return:
    """
    rows = len(data)
    cols = len(column_list)
    label = pd.DataFrame(np.zeros((rows,cols)))
    # print("column_list: ", column_list)
    # print("label: ", label.columns)
    label.columns = column_list
    for i in range(len(data)):
        for column in column_list:
            try:
                if data.loc[i, column] > 0:
                    label.loc[i, column] = 1
            except:
                continue
    return label


if __name__ == "__main__":
    # file_name = './data/weibo_train_data.txt'
    # df_1 = process_data(file_name,type='train')
    # print(df_1.head())
    test_file_name = './data/weibo_predict_data.txt'
    test_data = process_data(test_file_name, type='test')
    print(test_data.head())

    # column_list = ['forward_count','comment_count','like_count']
    # for column in column_list:
    #     data_column_1, residual = get_binary(df_1,column)
    #     print((data_column_1, residual))
    #
    #
    # label = get_num(df_1, column_list)
    # print(len(label))
