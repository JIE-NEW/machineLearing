#!/user/bin/python3
# -*- coding:utf-8 -*-

import csv
import numpy as np

_author_='WSQ'

#CSV文件存放地址
path1= r'G:/practice/Data/AdBoostData.csv'

#读取第n行的数据
def readNRow(n,path):
    csvFile = open(path,'r')
    reader = csv.reader(csvFile)

    feature = []
    for i,rows in enumerate(reader):
        if i==n:    
            feature = list(map(int,rows))
        else:
            pass
    csvFile.close()
    return feature      

#错误率统计函数

#阈值t,阈值正负，特征a的list,真实标签Y,权值list，返回分类结果的list
def createListOfTrue(t,flag,featurelist,labely,dataweight):
    #生成样本的预测标签list
    typeList = []
    n=len(featurelist)
    if flag==1:
        for i in range(n):
            if featurelist[i]<=t:
                typeList.append(-1)
            else:typeList.append(1)
    else:
        for i in range(n):
            if featurelist[i] > t:
                typeList.append(-1)
            else:typeList.append(1)

    #比较样本的标签与预测标签的差别
    #获取分类正误情况列表
    tfList=[]
    for (i1,i2) in zip(labely,typeList):
        #print(i1,i2)
        if i1==i2:
            tfList.append(0)
        else:tfList.append(1)
    #print('tfList:',tfList)
    return tfList

#分类误差率
#阈值t,特征a的list,真实标签Y,返回分类错误的个数
def ratioOfErrors(t,featurelist,labely,dataweight,flag):
    #生成样本的预测标签list
    #print(type(flag),type(1))
    typeList = []
    n=len(featurelist)
    if flag == 1:
        for i in range(n):
            if featurelist[i]<=t:
                typeList.append(-1)
            else:typeList.append(1)
    else:
        for i in range(n):
            if featurelist[i]>t:
                typeList.append(-1)
            else:typeList.append(1)

    #比较样本的标签与预测标签的差别
    #获取分类正误情况列表
    tfList=[]

    for (i1,i2) in zip(labely,typeList):
        #print(i1,i2)
        if i1==i2:
            tfList.append(0)
        else:tfList.append(1)
    ratio = list(map(lambda x:x[0]*x[1],zip(dataweight,tfList)))   
    #print('分类误差率：',ratio)
    sum_ratio=sum(ratio)
    return sum_ratio


#决策树桩 input: 特征x，真实标签Y，学习步长L,样本权值分布W
def dTree(x,Y,L,W):
    #设定步长和错误率
    minErr = 9999
    tbest=999
    _pm=10
    pm=10
    x_feature=readNRow(x,path1)
    for tvalue in np.arange(0, 3, L):
        #print(tvalue)
        #numErr = numberOfErrors(tvalue,x_feature,Y,W)  考虑在此处加上正负号
        numRatio_1 = ratioOfErrors(tvalue,x_feature,Y,W,1)
        numRatio_2 = ratioOfErrors(tvalue,x_feature,Y,W,-1)
       # print('分类误差率:',numRatio,'\n')
        if numRatio_1 < minErr:
            minErr = numRatio_1
            tbest = tvalue
            pm=1
        if numRatio_2 < minErr:
            minErr = numRatio_2
            tbest = tvalue
            pm=-1
    #print(u'正负pm',pm)

    #返回阈值,和最小分类误差率,分类器正负
    return tbest,minErr,pm

#简单分类器 树桩 特征序号，与特征对应的阈值
def rootDivide(x,t,flag):
    featurelist=readNRow(x,path1)
    #生成样本的预测标签list
    typeList = []
    n=len(featurelist)
    if flag==1:
        for i in range(n):
            if featurelist[i]<=t:
                typeList.append(-1)
            else:typeList.append(1)
    else:
        for i in range(n):
            if featurelist[i]>t:
                typeList.append(-1)
            else:typeList.append(1)  
    return typeList
    

#逻辑斯提函数
def sigmoid(inX):
    #return 1.0/(1+np.exp(-inX))
    oneList1 = np.ones(len(inX))
    
    for i in range(len(inX)):
        if inX[i]<=0:
            oneList1[i]=-1
        else:oneList1[i]=1
        
    #print(oneList)      
    return oneList1


#AdaBoost 算法

#假设特征有n维，我们针对每一维特征求一个分类器，
# 选取这些分类器中分类误差率最低的分类器作为本轮的分类器，将其特征坐标与分类器一起存入G(x)中。 
def AdaBoost():
    #根据训练数据个数初始化权值分布
    #弱分类器正负、系数 及相应特征位置,和阈值
    plus_minus=[]
    alpha=[]
    g_feature=[]
    test=[]
    final_pm=[]
    
    #获取标签
    true_label=readNRow(4,path1)

    #生成权值list
    weight=[]
    num_weight=len(true_label)
    for i in range(num_weight):
        weight.append(1/num_weight)


    
    run_flag=9
    while(run_flag):
    
        #最小分类误差率
        minError=[999.0,999.0,999.0,999.0]
        #判别阈值
        t_minError=[999,999,999,999]
        #分类器正负
        plus_minus=[10,10,10,10]
        #分类误差率
        #feature_minError=10

        for i in range(1,4):
            #print('i:',i)
            t_minError[i],minError[i],plus_minus[i]=dTree(i,true_label,1,weight)

        #获取取得的最小分分类误差率
        feature_minError=min(minError)
        #print('最小分类误差率:',feature_minError)

        #获取取得最小分类误差率的特征是哪一个
        index_feature_minError=minError.index(feature_minError)

        #获取阈值
        new_t=t_minError[index_feature_minError]
        test.append(new_t)

        #获取阈值对应正负
        new_pm=plus_minus[index_feature_minError]
        final_pm.append(new_pm)

        #获取是哪一个特征
        g_feature.append(index_feature_minError)

        #计算G(X)的系数
        gm=0.5*np.log((1-feature_minError)/feature_minError)
        alpha.append(gm)
        
        #print(alpha,g_feature)
        #print('test:',test)
        #print('g_feature:',g_feature)
        print('系数:',alpha)
        #print('final_pm:',final_pm)


        #更新训练权值
        #重新计算分类正确的结果
        trueList1=createListOfTrue(new_t,new_pm,readNRow(index_feature_minError,path1),true_label,weight)
        #先生成未归一化权值
        #print(trueList1)
        oneList = np.ones(num_weight)
        #print('oneList:',oneList)
        #print(oneList) #此处是出错的地方
        new_weight=list(map(lambda x:(np.exp(-gm)*(x[0]-x[1])+np.exp(gm)*x[1])*x[2],zip(oneList,trueList1,weight)))
        #print('new_weight:',new_weight)
        #计算Z
        Z=sum(new_weight)
        #print(Z)
        weight=list(map(lambda x:x[0]/Z,zip(new_weight)))
        #print('weight:',weight)
        #归一化权值
        
        #生成判断依据，循环结束条件
        final_list=np.zeros(len(weight))
        for i in range(len(alpha)):
            #生成权值list,然后相加得到结果，再进行二值化
            #具体是哪一个特征被使用需要看记录的是谁

            finlist=rootDivide(g_feature[i],test[i],final_pm[i])
            #print('单步分类结果:',finlist)
            alpha_t=alpha[i]
            final_list=list(map(lambda x:x[0]+alpha_t*x[1],zip(final_list,finlist)))
        #print('final_list:',final_list)
        h=sigmoid(final_list)
        

        run_flag=sum(list(map(lambda x:abs(x[0]-x[1]),zip(h,true_label))))
        print('错误率:',run_flag/10,'\n')

#调用函数
AdaBoost()