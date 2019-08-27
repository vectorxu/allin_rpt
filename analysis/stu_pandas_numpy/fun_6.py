#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv('../../Datasets/BL-Flickr-Images-Book.csv')

#显示全部列
pd.set_option('display.max_columns', None)
# #显示全部行
# pd.set_option('display.max_rows', None)
#设置每一行最大显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# 重新摆放列位置
# columns = ['','']
# df = pd.DataFrame(df, columns = columns)

# 移除一个DataFrame中不想要的行或列
to_drop = ['Edition Statement',
           'Corporate Author',
           'Corporate Contributors',
           'Former owner',
           'Engraver',
           'Contributors',
           'Issuance type',
           'Shelfmarks']

# axis=1代表列，0代表行
df.drop(to_drop, inplace=True, axis=1)
# 也可以直接用cloumns制定列删除
# df.drop(columns=to_drop, inplace=True)

# print(df.head())


# 查看Identifier是否是唯一索引
# print(df['Identifier'].is_unique)
df.set_index('Identifier')
# print(df.head())
# print(df.loc[206])
# # 基于位置
# print(df.iloc[0])
#
# print(df.get_dtype_counts())


# print(df.loc[1905:, 'Date of Publication'].head(10))


# print(df['Date of Publication'].head(10))
#
# extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
# # print(extr.head(10))
#
# print(df.dtypes)
# df['Date of Publication'] = pd.to_numeric(extr)
# print(df.dtypes)
#
# # 计算缺失值的比例
# print(df['Date of Publication'].isnull().sum() / len(df))
#
# pub = df['Place of Publication']
# london = pub.str.contains('London')
# oxford = pub.str.contains('Oxford')
# df['Place of Publication'] = np.where(london, 'London',np.where(oxford, 'Oxford', pub.str.replace('-', ' ')))
# print(df['Place of Publication'])


university_towns = []
with open('../../Datasets/university_towns.txt') as file:
    for line in file:
        if '[edit]' in line:
            # Remember this `state` until the next is found
            state = line
        else:
            # Otherwise, we have a city; keep `state` as last-seen

            university_towns.append((state, line))

# print(university_towns[:5])


towns_df = pd.DataFrame(university_towns,columns=['State','RegionName'])



def get_citystate(item):
    if ' (' in item:
        return item[:item.find(' (')]
    elif '[' in item:
        return item[:item.find('[')]
    else:
        return item


towns_df = towns_df.applymap(get_citystate)


olympics_df = pd.read_csv('../../Datasets/olympics.csv')

# 移除第一行
olympics_df = pd.read_csv('../../Datasets/olympics.csv', header=1)
print(olympics_df.head())

new_names =  {'Unnamed: 0': 'Country',
              '? Summer': 'Summer Olympics',
              '01 !': 'Gold',
              '02 !': 'Silver',
              '03 !': 'Bronze',
              '? Winter': 'Winter Olympics',
              '01 !.1': 'Gold.1',
              '02 !.1': 'Silver.1',
              '03 !.1': 'Bronze.1',
              '? Games': '# Games',
              '01 !.2': 'Gold.2',
              '02 !.2': 'Silver.2',
              '03 !.2': 'Bronze.2'}
# 重命名列名字
olympics_df.rename(columns=new_names, inplace=True)

print(olympics_df.head())















