# coding=utf-8
import os, sys
import requests
import random
import numpy as np
import pandas as pd

# https://www.jianshu.com/p/7414364992e4
def demo():
	d1 = pd.Series(np.arange(5,10),index=['ha1','jim','t','kite','why'])
	print(d1)
	d2 = pd.DataFrame(np.arange(5,11).reshape(2,3))
	print(d2)
	print('------')
	print(d2[:1])
	print('------')
	print(d2[:][0])

def readDemo(filepath_or_buffer):
	# sep: 分隔符
	# names: 列索引的名字
	# usecols: 指定读取的列名
	# 返回的类型: DataFrame
	# filepath_or_buffer : 文件路径(本地路径或url路径)
	result = pandas.read_csv(filepath_or_buffer, sep=",", names=None, usecols = None)
	# 打印前5个
	print("-->前5个:")
	print(result.head(5))
	# 打印后5个
	print("-->后5个:")
	print(result.tail(5))
	# 打印描述信息(实验中好用)
	print("-->描述信息:")
	print(result.describe())

demo()