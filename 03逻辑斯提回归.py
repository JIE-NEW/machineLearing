#!/usr/bin/python3
# -*- coding:utf-8 -*-

#读取数据，存入矩阵

import codecs

from datetime import datetime
import numpy as np
import os  
import csv  
import sys
import re
import pandas as pd
__author__ = 'WSQ'  


#生成要训练的标签
def creatLabel(labely,L):
    #for 循环遍历，!=n,设置为0
    print('训练数字：',L)
    n=labely.shape
    #print(n[0])
    labelY = np.ones((n[0],1))
    i=0
    for m in labely:
        #print(m,type(m),'----------------------')
        #print(np.float64(L))
        if m != np.float64(L):
            labelY[i]=0  #再改
        elif m==np.float64(L):
            labelY[i]=1
        i=i+1
    print('训练标签',labelY.T)
    return labelY




#逻辑斯提函数
def sigmoid(inX):
    #判断正负修改函数
    '''if inX >= 0:
        return 1.0/(1+np.exp(-inX))
    elif inX < 0:
        return np.exp(inX)/(1+np.exp(inX))
    ''' 
    return 1.0/(1+np.exp(-inX))   
        
        
    

#测试函数
def  test(dataMatIn,weights,Label):
    n=Label.shape
    r=0
    labelY = np.ones((n[0],1))
    h=sigmoid(np.dot(dataMatIn,weights))

    #判断h中每一个数据的大小,得到预测结果labelY
    i=0
    for m in h:
            #print(m,type(m),'----------------------')
        #print(np.float64(L))
        if m >= np.float64(0.5):
            labelY[i]=1  #再改
        elif m < np.float64(0.5):
            labelY[i]=0
        i=i+1

    print('预测标签',labelY.T)
    #给出正确率
    for (i1,i2) in zip(Label,labelY):
        if i1==i2:
            r=r+1
    print('r:',r,'n:',n[0])
    print('准确率：',r/n[0])

    return r/n[0]
            

    




#梯度上升函数
def gradAscent(dataMatIn,classLabels):
    mn=dataMatIn.shape
    m=mn[0]
    n=mn[1]
    #print(mn,m,n)
    alpha = 0.002
    maxCycles = 500
    weights = np.ones((n,1))
    flag=10
    time=1
    while flag >= 0.0003:
        #print('Run Times:',time,' Start time:',datetime.now())
        h=sigmoid(np.dot(dataMatIn,weights))
        error = (h-classLabels)
        transit = np.dot(alpha,dataMatIn.transpose())
        detW = np.dot(transit,error)
        weights = weights - detW
        flag1 = np.dot(detW.T,detW)
        time = time+1
        #print(flag)
        flag=sum(flag1[0,:])
        #print('flag',flag)
    print('迭代词数:',time,'学习率=',alpha)
    return weights


    
#txtFile = open(path + '/' + filename,'r',encoding="utf8")
martixFile= open('G:/厦门大学/统计机器学习/03作业/softmax多分类-数据集/testData.csv','rb')
labelFile= open('G:/厦门大学/统计机器学习/03作业/softmax多分类-数据集/label.csv','rb')

A=np.loadtxt(martixFile,delimiter=',',skiprows=0)
sLabel=np.loadtxt(labelFile,delimiter=',',skiprows=0)

trainLabel = creatLabel(sLabel,0)
W=gradAscent(A,trainLabel)
     
#print(W.T) 

labelFile.close()
martixFile.close()


#准确率测试
testMartixFile= open('G:/厦门大学/统计机器学习/03作业/testData.csv','rb')
teatLabelFile= open('G:/厦门大学/统计机器学习/03作业/label.csv','rb')
testA=np.loadtxt(testMartixFile,delimiter=',',skiprows=0)
testsLabel=np.loadtxt(teatLabelFile,delimiter=',',skiprows=0)

testLabel = creatLabel(testsLabel,0)

test(testA,W,testLabel)