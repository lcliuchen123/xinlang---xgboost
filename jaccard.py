
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy.linalg import  norm

# 1.利用jaccard系数计算文本相似度，利用词袋向量，分词或者不分词
def jaccard_sim(s1,s2):
    """s1,s2必须以空格隔开"""
    s1 = ' '.join(list(s1))
    s2 = ' '.join(list(s2))
    corpus = [s1,s2]
    bg = CountVectorizer(tokenizer=lambda x: x.split())
    vectors = bg.fit_transform(corpus).toarray()
    print("vectors: ", vectors)
    numerator = np.sum(np.min(vectors,axis=0))   #取每列的最小值相加，交集
    print(numerator)
    denominator = np.sum(np.max(vectors,axis=0)) #取每列的最大值相加，并集
    print(denominator)
    sim = 1.0* numerator/denominator

    return sim

#2.TF-IDF相似度
def tfidf_sim(s1,s2):
    s1 = ' '.join(list(s1))
    s2 = ' '.join(list(s2))
    corpus = [s1,s2]
    tf = TfidfVectorizer(tokenizer= lambda x : x.split())
    vectors = tf.fit_transform(corpus).toarray()
    tf_sim = np.dot(vectors[0],vectors[1]) / (norm(vectors[0])* norm(vectors[1]))

    return tf_sim

# 3.编辑距离
def  edit_ditance(s1,s2):
    length_1 = len(s1)
    length_2 = len(s2)
    result = [[i+j for j in range(length_2+1)] for i in range(length_1+1)]

    for i in range(1,length_1+1):
        for j in range(1,length_2+1):
            if s1[i-1] == s2[j-1]:
                d = 0
            else:
                d = 1
            result[i][j] = min(result[i-1][j]+1,result[i-1][j-1]+d,result[i][j-1]+1)

    return result[length_1][length_2]

if __name__ == "__main__":
    s1 = "垃圾清理垃圾清理清理垃圾"
    s2 = "帮我清理一下手机里的垃圾"
    bg_sim = jaccard_sim(s1,s2)
    tf_sim = tfidf_sim(s1,s2)
    edis = edit_ditance(s1,s2)
    print("sim: ",bg_sim)
    print("tf_sim: ",tf_sim)
    print("edis: ", edis)
