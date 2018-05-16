#!/user/bin/python3
# -*- coding:utf-8 -*-

import csv
import json
import re
import numpy as np

_author_='WSQ'


#注册类型
UserType=[0,0,0,0,0,0,0,0,0,0,0,0]

#性别
Gender=[0,0]

#地点字典
Address={}

userList = open('C:/Users/wangshqiang/Desktop/workspace/gloabl/AllUidV3.txt','r',encoding="utf8")

#计数
i=0
for line in userList.readlines():
    
    fline = line.split()
    #print(fline) 


    #获得用户的注册类型
    if fline[2]=='Y':
        UserType[int(fline[3])]=UserType[int(fline[3])]+1

    #获得用户的地域分布
    if Address.get(fline[9],0)==0:
        Address[fline[9]]=1
    else:
        Address[fline[9]]=Address[fline[9]]+1

    #获得性别比例
    if fline[4] == 'f':
        Gender[0]= Gender[0] + 1
    elif fline[4]== 'm':
        Gender[1]= Gender[1] + 1



    #writeFile.write(fline+'\n')
    
    i= i + 1
    '''if i==1168314:
        break'''
print('总数人数：',i)
print('性别: 男：',Gender[1],'女：',Gender[0])
print('地域分布',sorted(Address.items(), key = lambda item:item[1]))
#print('地域分布',Address.sort(lambda a,b :-cmp(a[1],b[1])))
print('用户类型：黄V:',UserType[0],' 政府：',UserType[1],' 企业：',UserType[2],' 媒体：',UserType[3],' 院校：',UserType[4],' 网站：',UserType[5],' 新浪：',UserType[7]) 
userList.close()

