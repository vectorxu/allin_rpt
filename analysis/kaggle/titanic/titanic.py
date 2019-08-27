#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入门分析项目

Kaggle作为公认的数据挖掘竞赛平台：泰坦尼克号生还者预测

https://mp.weixin.qq.com/s?__biz=MzUzODYwMDAzNA==&mid=2247484422&idx=1&sn=d3b2433d77c5d334208e3be112d036a2&chksm=fad4730bcda3fa1d608a9306a595db7b22d53f7d0fc963b22e0e3b118bb860f34e11f5eee2de#rd
"""



"""
导入数据
"""
import pandas as pd
import numpy as np

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
df = data_train.append(data_test)