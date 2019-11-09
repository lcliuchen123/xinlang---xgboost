# 利用xgboost 预测转发数、点赞数和互动数

## 数据字段解释
### uid:用户id
### mid：博文标记
### time：发博时间；
### forward_count：博文发表一周后的转发数
### comment_count：博文发表一周后的评论数
### like_count：博文发表一周后的赞数
### content：博文内容

## 第一步：特征工程
实际预测选取了10个指标,用*标记
1. 查看转发数，评论数，点赞数的分布情况以及它们之间的数值关系。

    划分类别：0表示为0,1表示大于0.
    对不同的类别采取不同的预测方式。
    
    共有八种类别（点赞数，转发数，评论数）：
        （0,0,0）,（1,0,0）,（0,1,0）, （0,0,1）
         (1,1,0),  (1,0,1),  (1,1,0),  (1,1,1)

2. 提取用户特征，观察每个用户发表的博文数量与转发数，评论数和点赞数之间的关系。

       用户的关注人数  未知
       粉丝数 未知
       注册时间 未知
       
       每个用户发表博文的数量 待定
       用户发表博文的平均评论数，最大评论数，最小评论数
       *平均转发数，最大转发数，最小转发数
       平均点赞数，最大点赞数，最小点赞数
       
       不同时间段的平均，最小，最大转发数，评论数和点赞数 （影响可能较小，待定）
   
3. 提取博文特征

    3.1 *博文发表时间
    
        是否是节假日
        是否是周末
        是否是高峰时间段
        
        高峰时段：8-12和20-24
        四个等级：
            非节假日或者周六日且非高峰时段标记为1，
            非节假日或者周六日的高峰时段标记为2，
            节假日或者周六日的非高峰时段标记为3，
            节假日或者周六日的高峰时段标记为4

    3.2 博文内容特征
    
        *是否包含@:  @可能是@别人，也可能是转发别人的微博
        *微博长度
        *高频词出现次数
        *是否包含url: #或者http://
        *是否包含表情：表情一般在[]中
        *是否包含标题：标题一般在【】中
        
        是否包含广告 待定
        情感分析 正向，中性和负向 待定
        是否涉及明星或者高热话题 待定
       
        
## 第二步：训练模型，预测博文发表一周后的转发数，评论数和点赞数
   ### 训练数据是一周后的转发数，评论数和点赞数，理论上是远远低于发表一天后的转发数，评论数和点赞数。
   ### 目前以泊松回归为基分类器，利用xgboost将转发数、评论数和点赞数分别进行预测
    1. 划分训练集，测试集   7:3
    2. 用户分类，博文分类   待定
    3. xgboost预测一周后的转发数，评论数，点赞数
        如何将转发数，评论数，点赞数结合在一块进行预测？  待定
                  对不同的类别采用不同的预测方式
                      如果只有一个不为0，就是只有一个因变量需要预测。
                      如果有两个不为0，就是需要预测两个因变量。
                      如果三个全部不为0，需要预测三个因变量。
                      三个因变量应该是相关的。
            算法设计：      待定
                    第一步：划分类别（8类），预测每个样本所属的类别。（该步可以省略，直接从第二步开始）
                    第二步：利用构建好的特征，分别对转发数，点赞数和评论数进行预测（xgboost）。
                    第三步：计算真实值与第二步预测值之间的误差，以误差值为虚拟变量，然后对点赞数，评论数和转发数进行建模。
                    第四步：利用两步之和作为最终预测值。
                      
        