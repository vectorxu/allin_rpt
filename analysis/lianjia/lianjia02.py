# -*- coding: utf-8 -*-
#!/usr/bin/env python

# https://segmentfault.com/a/1190000015440560

import os
import sys

# 科学计算包numpy,pandas,
# 可视化matplotlib,seaborn,
# 以及机器学习包sklearn


import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib as mpl
from IPython.display import display
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})

'''
解决中文方块问题
'''
## ---- 解决中文方块问题

plt.rcParams['font.family'] = ['Arial Unicode MS'] #解决中文显示方块问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

## ------

# 检查Python版本
from sys import version_info
if version_info.major != 3:
    raise Exception('请使用Python 3 来完成此项目')


# 导入二手房数据
lianjia_df = pd.read_csv('lianjia.csv')
display(lianjia_df.head(n=2))

# 检查缺失值情况
#lianjia_df.info()

# 数据描述
# print(lianjia_df.describe())


# 添加新特征房屋均价
df = lianjia_df.copy()
df['PerPrice'] = lianjia_df['Price']/lianjia_df['Size']

# 重新摆放列位置
columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Elevator', 'Direction', 'Renovation', 'PerPrice', 'Price']
df = pd.DataFrame(df, columns = columns)

# 重新审视数据集
display(df.head(n=2))



"""
特征工程
"""
# 移除结构类型异常值和房屋大小异常值
df = df[(df['Layout']!='叠拼别墅')&(df['Size']<1000)]

# 去掉错误数据“南北”，因为爬虫过程中一些信息位置为空，导致“Direction”的特征出现在这里，需要清除或替换
df['Renovation'] = df.loc[(df['Renovation'] != '南北'), 'Renovation']

# 由于存在个别类型错误，如简装和精装，特征值错位，故需要移除
df['Elevator'] = df.loc[(df['Elevator'] == '有电梯')|(df['Elevator'] == '无电梯'), 'Elevator']

# 填补Elevator缺失值
df.loc[(df['Floor']>6)&(df['Elevator'].isnull()), 'Elevator'] = '有电梯'
df.loc[(df['Floor']<=6)&(df['Elevator'].isnull()), 'Elevator'] = '无电梯'


# print(df['Layout'].value_counts())

# 只考虑“室”和“厅”，将其它少数“房间”和“卫”移除
# df = df.loc[df['Layout'].str.extract('^\d(.*?)\d.*?') == '室']

# print(df['Layout'].value_counts())

# 提取“室”和“厅”创建新特征
df['Layout_room_num'] = df['Layout'].str.extract('(^\d).*', expand=False).astype('int64')
df['Layout_hall_num'] = df['Layout'].str.extract('^\d.*?(\d).*', expand=False).astype('int64')

# print(df['Layout_room_num'].value_counts())
# print(df['Layout_hall_num'].value_counts())

# 按中位数对“Year”特征进行分箱
df['Year'] = pd.qcut(df['Year'],8).astype('object')

print(df['Year'].value_counts())

# 对“Direction”特征
d_list_one = ['东','西','南','北']
d_list_two = ['东西','东南','东北','西南','西北','南北']
d_list_three = ['东西南','东西北','东南北','西南北']
d_list_four = ['东西南北']
# df['Direction'] = df['Direction'].apply(direct_func)
# df = df.loc[(df['Direction']!='no')&(df['Direction']!='nan')]

# 根据已有特征创建新特征
df['Layout_total_num'] = df['Layout_room_num'] + df['Layout_hall_num']
df['Size_room_ratio'] = df['Size']/df['Layout_total_num']

# 删除无用特征
df = df.drop(['Layout','PerPrice','Garden'],axis=1)

# 对于object特征进行onehot编码
# df,df_cat = one_hot_encoder(df)













