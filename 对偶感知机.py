#!/usr/bin/python
import random
import numpy
import matplotlib.pyplot as plt


#全局变量
x1=[]
x2=[]
y=[]
w=[0,0]



#生成正例点
for i in range(0,50):
    x1.append(random.uniform(2,10))
    #print('0-60',i)
for i in range(0,50):
    x2.append(random.uniform(2,10))
    #print('60-100',i)

#生成反例点
for i in range(50,100):
    x1.append(random.uniform(-6,0))
    
for i in range(50,100):
    x2.append(random.uniform(-2,2))

#生成标签
for i in range(0,50):
    y.append(1)
   
for i in range(50,100):
    y.append(-1)
#print(y)
    
'''
print(x)
print('-'*10)
print(y)
'''

#感知机函数
def Perceptron(x1,x2,y,a):
    w=[0,0]
    b=0
    flag=1
    s=0

    while(flag):
        #判断误分类点并更新权值
        for s in range(100):
            while y[s]*(w[0]*x1[s] + w[1]*x2[s] + b) <= 0:
                #print("while 循环",s)
                w1=w[0] + a*x1[s]*y[s]
                w2=w[1] + a*x2[s]*y[s]
                b1=b+a*y[s]
                #赋值给w,b
                w=[w1,w2]
                b=b1
                #print(w,b)
        

        print("-----------一轮----------------")
        #运算结束标志判断
        flag=0
        for i in range(100):
            #print("运算结束标志",i)
            if y[i]*(w[0]*x1[i] + w[1]*x2[i] + b) <= 0:
                flag=1
                s=i
            if flag==1:
                break


    #直线绘制
    final=[-1*w[0]/w[1],-1*b/w[1]]
    func=numpy.poly1d(final)
    xl=numpy.linspace(-2,4,10)
    #print(xl)
    yl=func(xl)


    #绘图
    fig = plt.figure(dpi=150, figsize=(20, 6))
    plt.grid(True)
    #读取数据
    plt.scatter(x1, x2,c='red', alpha=1, marker='.', label='感知机')
    plt.plot(xl, yl, c='blue',marker='.',alpha=0.5,label='Line')
    plt.show()
    plt.close()      

Perceptron(x1,x2,y,1)



