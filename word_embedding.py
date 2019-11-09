#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
from jieba.analyse import *
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

#1.把.xml.bz2转化为.txt文件
input_file = './zhwiki-latest-pages-articles.xml.bz2'
# input_file = "C:/Users/root/PycharmProjects/word2vec/zhwiki-20181020-pages-articles.xml.bz2"
wiki = WikiCorpus(input_file, lemmatize=False, dictionary={})
count = 0
with open("./wiki.txt",'w',encoding="utf-8") as f:
# with open("C:/Users/root/PycharmProjects/word2vec/wiki.zh.text.txt",'w',encoding="utf-8") as f:
    for line in wiki.get_texts():
        print(line)
        line=' '.join(line)+"\n"
        f.write(line)
        count += 1
print("the %d articles have been writed" % count)

#2.繁体转化为简体（命令行）
#opencc -i wiki.zh.text.txt -o test.txt -c t2s.json

# 3.jieba分词，去除停词
# 3.1加载停词表
with open("stopwords.txt",'r',encoding="utf8") as f:
    stoplist=[]
    for line in f.readlines():
        line=line.strip("\n")
        stoplist.append(line)
# 3.2 同时进行分词和去除停词
with open("stopwords.txt",'r',encoding="utf8") as f:
    stoplist=[]
    for line in f.readlines():
         line=line.strip("\n")
         stoplist.append(line)
# print(stoplist)

#TF-IDF提取高频词
for keyword, weight in extract_tags('文本', withWeight=True, topK=100):
    print('%s %s' % (keyword, weight))

# 分词后的维基百科数据
target = open("zh.seg-1.3gg.txt", 'w',encoding="utf8")
print ('open files')
line_num=1
with open("test.txt","r",encoding="utf8") as f1:
    new_line=" "
    line = f1.readline()
    while line:
        print('---- processing ', line_num, ' article----------------')
        line_seg = " ".join(jieba.cut(line))
        print(line_seg)
        for word in line_seg:
            if word not in stoplist:
                new_line += word
        print(new_line)
        target.writelines(line_seg)
        line_num = line_num + 1
        line = f1.readline()
target.close()
print(line_num)

#4.建立模型并保存
inp="zh.seg-1.3gg.txt"
outp1="wiki.zh.text.vector.model"
outp2 = "wiki.zh.text.vector"
#LineSentence预处理大文件
model = Word2Vec(LineSentence(inp), size=300, window=5, min_count=5,
                 workers=multiprocessing.cpu_count())
model.save(outp1)
model.wv.save_word2vec_format(outp2, binary=False)

#5.测试
en_wiki_word2vec_model = Word2Vec.load('wiki.zh.text.vector.model')#wiki.zh.text.model：模型名字
testwords = ['苹果','数学','学术','白痴','篮球']
for i in range(5):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print (testwords[i])
    vector = en_wiki_word2vec_model[testwords[i]]
    print(vector.shape)
    print (res)