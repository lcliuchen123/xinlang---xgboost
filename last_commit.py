
import pandas as pd

# 按照天池大赛格式提交结果
forward_count = pd.read_table('./result/forward_count_result.txt',quoting=3)
comment_count = pd.read_table('./result/comment_count_result.txt',quoting=3)
like_count = pd.read_table('./result/like_count_result.txt',quoting=3)
predict_data = pd.read_table('./data/weibo_predict_data.txt',usecols=[0,1],names=['uid','mid'],quoting=3)
predict_data['forward_count']  = forward_count['forward_count'].apply(lambda x: round(x))
predict_data['comment_count']  = comment_count['comment_count'].apply(lambda x: round(x))
predict_data['like_count']  = like_count['like_count'].apply(lambda x: round(x))
predict_data.to_csv('weibo_result_data.txt',sep='\t', index=False)

