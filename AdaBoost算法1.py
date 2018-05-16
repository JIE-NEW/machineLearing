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

    #获取分类正误情况列表
    tfList=[]
    for (i1,i2) in zip(labely,typeList):
        #print(i1,i2)
        if i1==i2:
            tfList.append(0)
        else:tfList.append(1)
    return tfList

#阈值t,特征a的list,真实标签Y,返回分类错误的个数
def ratioOfErrors(t,featurelist,labely,dataweight,flag):
    #生成样本的预测标签list
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
    #获取分类正误情况列表
    tfList=[]

    for (i1,i2) in zip(labely,typeList):
        #print(i1,i2)
        if i1==i2:
            tfList.append(0)
        else:tfList.append(1)
    ratio = list(map(lambda x:x[0]*x[1],zip(dataweight,tfList)))   
    #print('分类误差率：',ratio)
    return sum(ratio)
#决策树桩 input: 特征x，真实标签Y，学习步长L,样本权值分布W
def dTree(x,Y,L,W):
    minErr = 9999
    tbest=999
    _pm=10
    pm=10
    x_feature=readNRow(x,path1)
    for tvalue in np.arange(0, 3, L):
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
    print(u'正负pm',pm)
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
    oneList = np.ones(len(inX))
    
    for i in range(len(inX)):
        if inX[i]<=0:
            oneList[i]=-1
        else:oneList[i]=1    
    return oneList
 
def AdaBoost():
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
    
    run_time=1
    while(run_time):
        #最小分类误差率
        minError=[999.0,999.0,999.0,999.0]
        #判别阈值
        t_minError=[999,999,999,999]
        #分类器正负
        plus_minus=[10,10,10,10]

        for i in range(1,4):
            t_minError[i],minError[i],plus_minus[i]=dTree(i,true_label,1,weight)
        feature_minError=min(minError)
        index_feature_minError=minError.index(feature_minError)
        new_t=t_minError[index_feature_minError]
        test.append(new_t)
        new_pm=plus_minus[index_feature_minError]
        final_pm.append(new_pm)
        g_feature.append(index_feature_minError)
        gm=0.5*np.log((1-feature_minError)/feature_minError)
        alpha.append(gm)

        #更新训练权值
        trueList1=createListOfTrue(new_t,new_pm,readNRow(index_feature_minError,path1),true_label,weight)
        oneList = np.ones(num_weight)
        new_weight=list(map(lambda x:np.exp(-gm)*(x[0]-x[1])+np.exp(gm)*x[1],zip(oneList,trueList1)))
        Z=sum(new_weight)
        weight=list(map(lambda x:x[0]/Z,zip(new_weight)))

        final_list=np.zeros(len(weight))
        for i in range(len(alpha)):
            finlist=rootDivide(g_feature[i],test[i],final_pm[i])
            alpha_t=alpha[i]
            final_list=list(map(lambda x:x[0]+alpha_t*x[1],zip(final_list,finlist)))
        h=sigmoid(final_list)
        run_time=sum(list(map(lambda x:(x[0]-x[1])*(x[0]-x[1]),zip(h,true_label))))

AdaBoost()