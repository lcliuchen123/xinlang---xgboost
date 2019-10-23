
import pandas as pd
import numpy as np
from datetime import datetime
import jieba
from jieba.analyse import *

# 构造特征工程
# 获取互动的最大值、最小值和平均值
def max_min_mean(train_data, pred_data, column_name='uid'):
    """
    :param train_data: 训练集数据，表示用户历史博文信息
    :param pred_data: 待预测的数据
    :param column_name: 用户id
    :return: 返回用户历史博文数据的最大值、最小值和均值
    """
    unique_user = train_data['uid'].unique()
    total_user = len(unique_user)
    print("总用户数：", total_user)
    grouped = train_data.groupby('uid')
    feature = ['min', 'max', 'mean', 'sum', 'count']
    min_max_hudong = grouped.agg({'forward_count': feature, 'comment_count': feature, 'like_count': feature})

    new_df = pd.merge(pred_data, min_max_hudong, on='uid')
    new_df.columns = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content',
                      'min_forward_num', 'max_forward_num', 'mean_forward_num', 'total_forward_num',
                      'count_forward_count',
                      'min_comment_num', 'max_comment_num', 'mean_comment_num', 'total_comment_num',
                      'count_comment_count',
                      'min_like_num', 'max_like_num', 'mean_like_num', 'total_like_num', 'count_like_count']
    return new_df

# 高峰时段：8-12和20-24
# 五个等级：非节假日或者周六日且非高峰时段标记为1，
# 非节假日或者周六日的高峰时段标记为2，
# 节假日或者周六日的非高峰时段标记为3，
# 节假日或者周六日的高峰时段标记为4

def get_weight_time(data, column_name='time'):
    weight_time = np.ones(len(data))
    #     导入日期和时间数据
    time = [date.split() for date in list(data[column_name])]
    time = pd.DataFrame(time, columns=['date', 'time'])

    #     增加月，日，周列,datetime.datetime.strptime()
    time['weekday'] = time['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() + 1)
    time['month'] = time['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month)
    time['day'] = time['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").day)

    #    设定权重
    holiday = ['01-01', '03-08', '04-05', '05-01', '06-20', '07-01', '08-01', '09-03', '09-18', '09-27', '10-01']
    for i in range(len(time)):
        if ((time.loc[i, 'weekday'] == 6) | (time.loc[i, 'weekday'] == 7)) | (time.loc[i, 'date'][5:] in holiday):
            if ((time.loc[i, 'time'] >= '08') & (time.loc[i, 'time'] < '12')) | (
                    (time.loc[i, 'time'] >= '20') & (time.loc[i, 'time'] < '24')):
                weight_time[i] = 4
            else:
                weight_time[i] = 3
        else:
            if ((time.loc[i, 'time'] >= '08') & (time.loc[i, 'time'] < '12')) | (
                    (time.loc[i, 'time'] >= '20') & (time.loc[i, 'time'] < '24')):
                weight_time[i] = 2
            else:
                weight_time[i] = 1

    data['weight_time'] = weight_time
    return data

def get_other_features(data):
    # 0-1分类变量
    content_length = data['content'].apply(lambda x: len(x))  # 微博长度
    is_aite = data['content'].apply(lambda x: '@' in x)  # 是否包含@，可能为转发或者@别人
    is_url = data['content'].apply(lambda x: (('#' in x) | ("http://" in x)))  # 是否包含URL或者#，#表示为连接
    is_theme = data['content'].apply(lambda x: (('【' in x) & ('】' in x)))  # 是否包含标题
    is_face = data['content'].apply(lambda x: (('[' in x) & (']' in x)))  # 是否包含表情或者图片，一般表情或者图片包含在[]中

    data['is_aite'] = is_aite * 1
    data['is_url'] = is_url * 1
    data['is_theme'] = is_theme * 1
    data['is_face'] = is_face * 1
    data['content_length'] = content_length

    return data

# 利用TF-DF提取互动数量较多的样本中的高频词
# 首先，同时去除停词和进行分词
# 获取停词表
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf8').readlines()]
    return stopwords

# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('F:\\研一上\\研创\\研创结果\\词典\\stoplist.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        word = word.strip()
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

# TFIDF   可以设置停词或者其他,参考https://blog.csdn.net/Dorisi_H_n_q/article/details/82114649
def count_words(line, top_words):
    """返回每条博文含有top500词的个数"""
    count = 0
    for word in line.split():
        if word in top_words:
            count += 1
    return count

def get_top_words(data, column_name, topk, percent=80):
    """
    按照分位数选择转发数、点赞数和评论数较多的博文，提取top500词。
    """
    a = np.array(data[column_name])
    per = float(np.percentile(a, percent))  # 80%分位数

    data['words'] = data['content'].apply(lambda x: seg_sentence(x))
    content = data.loc[data[column_name] > per, 'words']

    text = ""
    for line in list(content):
        text += line
        text += ' '
    top_words = {}
    for keyword, weight in extract_tags(text, withWeight=True, topK=topk):
        print('%s %s' % (keyword, weight))
        top_words[keyword] = top_words.get(keyword, weight)

    topwords = list(top_words.keys())
    data['word_count'] = data['words'].apply(lambda x: count_words(x, topwords))
    result = pd.DataFrame(data, columns=['words', 'word_count'])
    return result

def topk_num(data):
    """获取点赞数、评论数和转发数所获取的高频词个数"""
    forward_result = get_top_words(data, 'forward_count', 500)
    forward_num = forward_result['word_count']
    comment_num = get_top_words(data, 'comment_count', 500)['word_count']
    like_num = get_top_words(data, 'like_count', 500)['word_count']

    data['forward_wordnum'] = np.array(forward_num)
    data['comment_wordnum'] = np.array(comment_num)
    data['like_wordnum'] = np.array(like_num)
    data['words'] = np.array(forward_result['words'])

    return data