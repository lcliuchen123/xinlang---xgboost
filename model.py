
#  make_scorer创造计分器评估模型，高的分意味着效果好。用于网格搜索
# 复制别人的代码，经常容易出现漏写，错写情况，deep参数和网格搜索时构造的mae_score参数

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import  KFold,train_test_split, GridSearchCV,RandomizedSearchCV
from cost_time import run_time



# 1.设置分类变量
@run_time
def astype_cate(data, cate_feature):
    """
    :param data: 数据
    :param cate_feature: 分类变量的列名
    :return:
    """
    try:
        for column in cate_feature:
            data[column] = data[column].astype('category').cat.codes
    except:
        print("error")

    return data

# 2.切分数据集
@run_time
def split_data(data, y_columns):
    """
    :param data:
    :param y_columns: 因变量的列名 -》链表
    :return: 切分后的训练集和测试集（所有特征）
    """
    y = pd.DataFrame(data, columns=y_columns)

    column_names = list(data.columns)
    for name in y_columns:
        if name in column_names:
            column_names.remove(name)
    x = pd.DataFrame(data, columns=column_names)
    seed = 7
    test_size = 0.3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    return x_train, x_test, y_train, y_test

# 3.分别获取转发数，评论数和点赞数的特征
@run_time
def get_single_feature(data, y_columns, xcolumn_names, ycolumn_names,type):
    """
    :param data:
    :param y_columns: 因变量的列名 -》链表
    :param xcolumn_names: 自变量的列名-》链表
    :param ycolumn_names: 需要预测的因变量-》Y
    :return: 返回单个因变量的训练集和测试集
    """
    try:
        if type == 'train':
            x_train, x_test, y_train, y_test = split_data(data, y_columns)
            single_xtrain = pd.DataFrame(x_train, columns=xcolumn_names)
            single_ytrain = pd.DataFrame(y_train, columns=ycolumn_names)
            single_xtest = pd.DataFrame(x_test, columns=xcolumn_names)
            single_ytest = pd.DataFrame(y_test, columns=ycolumn_names)

            return single_xtrain, single_ytrain, single_xtest, single_ytest
        else:
            single_xtrain = pd.DataFrame(data, columns=xcolumn_names)
            return single_xtrain
    except:
        print('the columns_list is error or the data is null')
        return None

# 4.构建xgboost模型
# 4.1构建绝对值损失函数
def xg_eval_mae(yhat, dtrain):
    """
    :param yhat: 因变量的预测值
    :param dtrain: xgboost对数据做的预处理
    :return: 平均绝对误差
    """
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.array(y), np.array(yhat))

# 4.2 构建模型性能评估函数
def mae_score(y_true, y_pred):
    return mean_absolute_error(np.array(y_true), np.array(y_pred))

# mae_scorer = make_scorer(mae_score,greater_is_better=False)

# 4.3构建XGBoostRegressor类
# xgboost 自定义了一个数据矩阵类 DMatrix，
# 会在训练开始时进行一遍预处理(具体指什么？？？？？？？？？），从而提高之后每次迭代的效率
class XGBoostRegressor(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'silent': 1, 'objective': 'count:poisson', 'seed': 0})

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                             feval=xg_eval_mae, maximize=False)

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)

    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)
        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=nfold, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
        return cv_rounds.iloc[-1, :]

    def plot_feature_importance(self):
        feat_imp = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Fature Importances')
        plt.ylabel("Feature Importance Score")

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

# 4.4调参，树的颗数和学习率调节较为费时，因此单独拿出来进行网格搜索
def grid_search(xgb_params_out, grid_params, xtrain, ytrain, function_name):
    for key in list(grid_params.keys()):
        if key in xgb_params_out:
            xgb_params_out.pop(key)

    mae_scorer = make_scorer(function_name, greater_is_better=False)
    grid = GridSearchCV(XGBoostRegressor(**xgb_params_out), param_grid=grid_params, cv=5, scoring=mae_scorer)
    grid.fit(xtrain, ytrain.values)

    return grid.cv_results_, grid.best_params_, grid.best_score_

@run_time
def get_best_xgb_params(xtrain, ytrain,eval_func='mae_score'):
    min_score = float('-inf')
    # 1.初始化模型
    xgb_params = {
        'seed': 0,
        'eta': 0.1,
        'colsample_bytree': 0.5,
        'silent': 1,
        'subsample': 0.5,
        'objective': 'count:poisson',
        'max_depth': 5,
        'min_child_weight': 3,
        'num_boost_round': 50
    }
    # 2.调整树的最大深度max_depth：一般在3-10之间。
    # 正则化参数min_child_weight：如果树分区中的实例权重小于定义的总和，则停止树构建过程。
    xgb_depth_weight = {'max_depth': list(range(3, 10)), 'min_child_weight': list((1, 3, 6))}
    first_grid_scores, first_best_params, first_best_score = \
        grid_search(xgb_params,xgb_depth_weight,xtrain, ytrain,eval_func)
    if first_best_score > min_score:
        min_score = first_best_score
        xgb_params.update(first_best_params)
        print('the params is updated')
    print("the best score of depth_and_weight is %f" % (float(first_best_score)))

    # 3.调节gamma去降低过拟合风险. gamma:损失下降多少才进行分裂
    xgb_gamma = {'gamma': [0.1 * i for i in range(0, 5)]}
    second_grid_scores, second_best_params, second_best_score = \
        grid_search(xgb_params,xgb_gamma, xtrain, ytrain,mae_score)
    if second_best_score > min_score:
        min_score = second_best_score
        xgb_params.update(second_best_params)
        print('the params is updated')
    print("the best score of gamma is %f" % (float(second_best_score)))

    # 4.调节样本采样方式 subsample 和 colsample_bytree
    xgb_sample_grid = {'subsample': [0.1 * i for i in range(6, 9)], 'colsample_bytree': [0.1 * i for i in range(6, 9)]}
    third_grid_scores, third_best_params, third_best_score = \
        grid_search(xgb_params, xgb_sample_grid, xtrain, ytrain,mae_score)
    if third_best_score > min_score:
        min_score = third_best_score
        xgb_params.update(third_best_params)
        print('the params is updated')
    print("the best score of subsample is %f" % (float(third_best_score)))
    print('the best score is: %f' % min_score)
    return xgb_params

@run_time
def get_eta_num(xgb_params,xtrain,ytrain,eval_func=mae_score):
    mae_scorer = make_scorer(eval_func, greater_is_better=False)
    best_result = []
    xgb_eta_tree = {'eta': [0.05 * i for i in range(0, 20)]}
    for num in list(range(50, 300, 50)):
        xgb_params.update({'num_boost_round': num})
        random_grid = RandomizedSearchCV(XGBoostRegressor(**xgb_params),
                                         param_distributions= xgb_eta_tree,cv=5,
                                         scoring=mae_scorer,n_iter=10,random_state=5)
        random_grid.fit(xtrain,ytrain.values)
        best_result.append([random_grid.best_estimator_,random_grid.cv_results_,
                            random_grid.best_params_, random_grid.best_score_])
    sorted(best_result, key=lambda x: x[-1], reverse=True)
    best_estimator,best_cv_result, best_params, best_score = best_result[0]

    return best_estimator,best_cv_result, best_params, best_score

# 返回最优的估计器
@run_time
def grid_eta_num(xgb_params,xtrain,ytrain):
    #     调节学习率eta和树个数num_boost_round，需要先设定树的棵数
    # 利用网格搜索需要消耗的时间过长
    best_result = []
    xgb_eta_tree = {'eta': [0.05 * i for i in range(0, 20)]}
    best_grid_scores, best_params, best_score = \
                grid_search(xgb_params, xgb_eta_tree, xtrain,ytrain, mae_score)
    # for num in list(range(50, 300, 50)):
    #     xgb_params.update({'num_boost_round': num})
    #     fourth_best_estimator,fourth_grid_scores, fourth_best_params, fourth_best_score = \
    #         grid_search(xgb_params, xgb_eta_tree, xtrain,ytrain, mae_score)
    #     best_result.append([fourth_best_estimator,fourth_grid_scores, fourth_best_params, fourth_best_score])
    #
    # sorted(best_result, key=lambda x: x[-1], reverse=True)
    # best_grid_scores, best_params, best_score = best_result[0]
    xgb_params.update(best_params)
    print("the best score of eta_num_boost is %f" % (float(best_score)))

    return best_grid_scores, best_params, best_score

if __name__ == "__main__":
    df = pd.read_table("F:\\process_data.txt", sep='\t', header=0)  # 默认情况下是header=0，表示把第一行作为列名
    cate_feature = ['weight_time', 'is_aite', 'is_url', 'is_theme', 'is_face']
    new_df = astype_cate(df,cate_feature)
    cate_feature.append('content_length')
    # y_columns:因变量 可能有多个，切分数据集
    y_columns = ['forward_count', 'comment_count', 'like_count']
    x_train, x_test, y_train, y_test = split_data(new_df, y_columns)
    # 获取转发数相关的训练集和测试集
    forward_xcolumn = ['min_forward_num', 'max_forward_num', 'mean_forward_num','forward_wordnum']
    forward_xcolumn.extend(cate_feature)
    forward_ycolumn = ['forward_count']
    single_xtrain, single_ytrain, single_xtest, single_ytest = \
        get_single_feature(new_df, y_columns, forward_xcolumn,forward_ycolumn,type = 'train')

    xgb_params = get_best_xgb_params(single_xtrain, single_ytrain)
    best_estimator,best_grid_scores, best_params, best_score = get_eta_num(xgb_params,single_xtrain,single_ytrain)
