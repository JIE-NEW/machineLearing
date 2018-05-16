#python 3
#author wsq
import random
import numpy
import matplotlib.pyplot as plt

label=[]
scroe=[]
X=[0]
Y=[0]

#生成标签数组label[100]
for i in range(0,60):
    label.append(1)
for i in range(60,100):
    label.append(-1)

#生成得分数组scroe[100]
for i in range(0,60):
    scroe.append(0.7)
    #print('0.7',i)
for i in range(60,100):
    scroe.append(0.3)
    #print('0.3',i)


#从score数组中随机挑选m+n 个位置修改为0-1之间的随机数
#利用Python中的randomw.sample()函数生成不重复的随机数 
def changeScroe(m,n):
    resultList=random.sample(range(0,60),m)
    for i in resultList:
        scroe[i]=random.random()
    resultList=random.sample(range(60,100),n)
    for i in resultList:
        scroe[i]=random.random()

    #按值排序，生成绘图数据X,Y
    for i in range(1,101):
        #print(i)       
        t = scroe.index(max(scroe))
        if i>0:
            f=i-1
            if label[t]== 1:
                X.append(X[f])
                y=Y[f]+1.0/60
                Y.append(y)
            elif label[t]== -1:             
                x=X[f]+1.0/40
                X.append(x)
                Y.append(Y[f])
            
            
        scroe[t]=-100
#绘图

    #print(X)
    #print(Y)
    # 下面是作图，在同一个画布上画两条曲线
    fig = plt.figure(dpi=150, figsize=(10, 10))
    plt.grid(True)

        #显示标题  
    title = 'ROC m='+ str(m) + ' n=' + str(n)  
    plt.title(title, fontsize=20)


    #读取数据
    plt.plot(X, Y, c='blue',marker='.',alpha=2,label='ROC')
    #plt.show()
    #保存图形
    plt.savefig('G:\厦门大学\机器学习/'+ title + '.png')
    plt.close()       


'''
        plt.xlabel("时刻", fontsize=16)
        #fig.autofmt_xdate()
        plt.ylabel("百分比", fontsize=16)
'''
        #plt.tick_params(axis='both', which='major', labelsize=16)

        #图例位置
        #plt.legend(loc=2, bbox_to_anchor=(1.0,1.0),borderaxespad = 0.1)
        #plt.legend(loc='upper right')



        #保存图形
        #plt.savefig(savepaths + '/'+ filename[:-4] + '.png')
        #画出图形

#改变m+n的值，画出10条FPR-TPR曲线
changeScroe(45,15)


    