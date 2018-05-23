#!/user/bin/ python3
# -*- coding:utf8 -*-

import numpy as np


_author_ = 'WSQ'

#初始化参数及全局参数
PList=[0.4,0.6,0.7]
ShowList=[1,1,0,1,0,0,1,0,1,1]


def ExpectFunction(plist,showlist):
    A1=list(map(lambda x:plist[0]*(np.power(plist[1],x[0]))*(np.power(1-plist[1],1-x[0])),zip(showlist)))
    A2=list(map(lambda x:(1-plist[0])*(np.power(plist[2],x[0]))*(np.power(1-plist[2],1-x[0])),zip(showlist)))
    Probability=list(map(lambda x:x[0]/(x[0]+x[1]),zip(A1,A2)))
    return Probability

def MaxFunction(plist1,showlist1,probability1):
    newPList=[0,0,0]
    P=sum(probability1)
    #更新pi值
    newPList[0]=0.1*P
    #更新B的概率
    P1=list(map(lambda x:x[0]*x[1],zip(probability1,showlist1)))
    newPList[1]=sum(P1)/P
    #更新C的概率
    P2=list(map(lambda x:(1-x[0])*x[1],zip(probability1,showlist1)))
    P3=list(map(lambda x:1-x[0],zip(probability1)))
    newPList[2]=sum(P2)/sum(P3)
    return newPList
i=10
#循环执行EM函数
while(i):
    Pro=ExpectFunction(PList,ShowList)
    newlist =MaxFunction(PList,ShowList,Pro)
    PList=newlist[:]
    print(PList)
    i=i-1
