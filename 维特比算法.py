#! user/bin/python3
# -*- coding:utf-8 -*-

import csv
import numpy as np

_author_='WSQ'

#模型参数

#状态转移概率矩阵
#A = [[0,1,0,0],[0.4,0,0.6,0],[0,0.4,0,0.6],[0,0,0.5,0.5]]
A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])

#观测概率矩阵
B=np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])

#初始状态概率向量
pi=np.array([0.2,0.4,0.4])

#观测序列，红色为0，白色为1
O=np.array([0,1,0])



'''
Red=[[0.5,0,0,0],[0,0.3,0,0],[0,0,0.6,0],[0,0,0,0.8]]
White=[[0.5,0,0,0],[0,0.7,0,0],[0,0,0.4,0],[0,0,0,0.2]]
'''


def marix2(A,B):
    return np.dot(A,B)
'''
#计算X1
X1=np.dot(pi,Red)
print('X1',X1)

#计算X2
X2=np.dot(marix2(X1,A),Red)
print('X2',X2)
'''
#最大路径存储
bigest_route= [[0 for i in range(3)] for i in range(3)]


#初始化 

X1=np.dot(pi,np.diag(B[:,O[0]]))
#print(X1)

#时刻t
for j in range(2):
    
    t_color=B[:,O[j+1]]
    #print(t_color,'\n')

    #每个时刻t对应的状态i
    i_status=[[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        #print(A[i])
        #print(X1[i])
        one_of_newprobality =list(map(lambda x:X1[i]*x[0]*x[1],zip(A[i],t_color)))
        i_status[i]=one_of_newprobality
        #print(one_of_newprobality,'---------\n',i_status)
    #分析记忆当前每一个状态的前一个最优状态
    #这一步
    for i in range(3):
        get_1=np.array(i_status)
        get = list(get_1[:,i])

        get_best= float(max(get))
        #print(type(get_best))
        get_best_index=get.index(get_best)
        bigest_route[i][j]=get_best_index
        X1[i]=get_best
print('newX1',X1)
print(bigest_route)
#根据结果输出最佳路径，还要正向输出
get_final_best=max(X1)
trace_route=X1.tolist().index(get_final_best)
#trace_route=get_final_index
print(trace_route+1)
for i in range(2):
    trace_route=bigest_route[trace_route][1-i]
    print(trace_route+1)

    
