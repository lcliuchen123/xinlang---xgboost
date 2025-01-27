
from model import *
from process_data import process_data
from create_feature import *
from cost_time import *
import numpy as np
import time
import os
import pickle

class Predict_data():
    def __init__(self,train_file_name,cate_feature,
                 column_list,y_columns_list,eval_func,test_file_name):
        self.cate_feature = cate_feature  #分类变量列表
        self.y_columns_list = y_columns_list        #因变量的列名列表
        self.process_column_list = column_list    #get_features需要处理的列名列表
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.eval_func = eval_func
        self.params = dict()
        self.all_estimator = dict()
        self.all_test_mae = dict()
        self.data = dict() #(xtrain,ytrain,xtest,ytest,x_predict)
        self.all_best_score = dict()
        self.top_word_dic = dict()

    @run_time
    def create_features(self):
        train_data = process_data(self.train_file_name,type='train')
        self.top_word_dic = get_top_words(train_data, self.y_columns_list, topk=500, percent=80)
        print('****************get the top words***********************')
        try:
            if self.train_file_name:
                self.train_features = get_features(train_data, train_data,self.y_columns_list,
                                                   self.top_word_dic,'train',self.process_column_list)
                print("*****the train features have been created*****")
                self.train_features.to_csv('./features/train.txt', sep='\t', index=False)
            if self.test_file_name:
                test_data = process_data(self.test_file_name,type='test')
                self.test_features = get_features(train_data, test_data,self.y_columns_list,
                                                  self.top_word_dic,'test', self.process_column_list)
                print("*****the test features have been created*****")
                self.test_features.to_csv('./features/test.txt', sep='\t', index=False)
        except:
            print("please input the file_name of train or test")

    @run_time
    def get_act_column_list(self,predict_type):
        try:
            if  predict_type == 'forward_count':
                act_xcolumn = ['min_forward_num', 'max_forward_num',
                               'mean_forward_num','forward_wordnum']
                act_ycolumn = ['forward_count']
                return act_xcolumn,act_ycolumn
            elif predict_type == 'comment_count':
                act_xcolumn = ['min_comment_num', 'max_comment_num',
                               'mean_forward_num', 'comment_wordnum']
                act_ycolumn = ['comment_count']
                return act_xcolumn, act_ycolumn
            elif predict_type == 'like_count':
                act_xcolumn = ['min_like_num', 'max_like_num', 'mean_like_num', 'like_wordnum']
                act_ycolumn = ['like_count']

                return act_xcolumn, act_ycolumn
        except:
            print("please input the type: forward_count or comment_count or like_count")

    @run_time
    def get_act_features(self,act_xcolumn,act_ycolumn):
        if 'train.txt' in os.listdir('.'):
            column = []
            column.extend(act_xcolumn)
            column.extend(self.cate_feature)
            column.extend(act_ycolumn)
            if 'content_length' not in column:
                column.append('content_length')
            self.train_features = pd.read_table('./features/train.txt',usecols=column,encoding='utf-8',quoting=3)
            print("the length of train_feature is %d" % len(self.train_features))
        if 'test.txt' in os.listdir('.'):
            self.test_features = pd.read_table('./features/test.txt',encoding='utf-8',quoting=3)
            print("the length of test_feature is %d" % len(self.test_features))

        train_features = astype_cate(self.train_features, self.cate_feature)
        test_features = astype_cate(self.test_features,self.cate_feature)
        if 'content_length' not in self.cate_feature:
            self.cate_feature.append('content_length')
        act_xcolumn.extend(self.cate_feature)
        xtrain, ytrain, xtest, ytest = \
                get_single_feature(train_features, self.y_columns_list, act_xcolumn, act_ycolumn,type='train')
        x_predict = \
                get_single_feature(test_features, self.y_columns_list, act_xcolumn, act_ycolumn,type='test')
        print('the length of predict_data is %d ' % len(x_predict))
        return xtrain, ytrain, xtest, ytest, x_predict

    @run_time
    def create_model(self,xtrain,ytrain):
        xgb_params = get_best_xgb_params(xtrain, ytrain,self.eval_func)
        best_estimator,best_cv_result, best_params, best_score = \
            get_eta_num(xgb_params, xtrain, ytrain,self.eval_func)

        return best_estimator, best_cv_result, best_params, best_score

    @run_time
    def get_result(self,col_name):
        act_xcolumn, act_ycolumn = self.get_act_column_list(col_name)
        xtrain, ytrain, xtest, ytest, x_predict = \
            self.get_act_features(act_xcolumn, act_ycolumn)
        print("***********create the train and test and predict*****************")
        self.data[act_ycolumn[0]] = (xtrain, ytrain, xtest, ytest, x_predict)
        print("*******************start the tuning params***************************")
        best_estimator, best_cv_result, best_params, best_score = self.create_model(xtrain,ytrain)
        print("*********************the best model have been created**********************")
        self.save_model(best_estimator,col_name)
        print("the %s model have been saved" % col_name)
        self.all_estimator[act_ycolumn[0]] = best_estimator
        self.params[act_ycolumn[0]] = best_params
        self.all_best_score[act_ycolumn[0]] = best_score
        print("********start the predict the test data*******************************************")
        ybeta_test = best_estimator.predict(xtest)
        test_mae = mae_score(ybeta_test,ytest)
        self.all_test_mae[act_ycolumn[0]] = test_mae
        print("test_mae: ", test_mae)
        print("********start the predict the predict data**********************************************")
        y_predict = best_estimator.predict(x_predict)
        result = x_predict
        result[act_ycolumn[0]] = np.array(y_predict)

        return result

    @run_time
    def save_model(self,model,col_name):
        file_name =  './model/%s_model' % col_name
        with open(file_name,'wb') as f:
            pickle.dump(model,f)

    @run_time
    def load_model(self,col_name):
        file_name = './model/%s_model' % col_name
        with open(file_name,'rb') as f:
            model = pickle.load(f)
        # model.predict()
        return model

    def get_all_result(self):
        for ycol_name in self.y_columns_list:
            result = self.get_result(ycol_name)
            file_name = './result/%s_result.txt' % ycol_name
            result.to_csv(file_name, sep='\t', index=False)
            print('the %s has been predicted!' % ycol_name)

if __name__ == "__main__":
    train_file_name = './data/weibo_train_data.txt'
    test_file_name = './data/weibo_predict_data.txt'
    cate_feature = ['weight_time', 'is_aite', 'is_url', 'is_theme', 'is_face'] # 分类变量列表
    y_columns_list =  ['forward_count', 'comment_count', 'like_count'] # 因变量的列名列表
    column_list =  ['uid','time']  # get_features需要处理的列名列表
    eval_func = mae_score
    predict_obj = Predict_data(train_file_name,cate_feature,column_list,
                               y_columns_list,eval_func,test_file_name)
    start_time = time.time()
    # predict_obj.create_features()
    predict_obj.get_all_result()
    end_time = time.time()
    # 直接预测需要54614秒，多线程仅需要29976s
    print('the cost of time is %f' % (end_time-start_time))#54614s

    # 预测案例
    model = predict_obj.load_model('forward_count')
    xtest = pd.DataFrame(np.ones((1,10)),columns=['min_forward_num', 'max_forward_num',
                                'mean_forward_num', 'forward_wordnum',
                                'weight_time', 'is_aite', 'is_url', 'is_theme',
                                'is_face', 'content_length'])
    xpredict = model.predict(xtest)
    print(xpredict)
