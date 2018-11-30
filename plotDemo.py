# coding=utf-8
import os, sys
import requests
import random
import numpy as np
import matplotlib.pyplot as plt

# 保证生成的图片在浏览器内显示
# %matplotlib inline
# 保证能正常显示中文(Mac)
plt.rcParams['font.family'] = ['Arial Unicode MS']

def drawDemo():
	x = np.arange(0,20,0.1)
	y = np.sin(x)
	y2 = np.cos(x)
	plt.plot(x,y)
	plt.plot(x,y2,'r--')
	plt.show()

def drawPoint():
	n = 1024
	X = np.random.normal(0,1,n)
	Y = np.random.normal(0,1,n)
	for i in range(1,10):
		plt.scatter(i, i)
	plt.title(u"散点图",color='red')
	plt.savefig("./p.png")
	plt.show()

def drawLine():
	# 模拟海南一天的温度变化
	# 生成x轴的24小时
	hainan_x = [h for h in range(0, 24)]
	# 生成y轴的温度随机值(15, 25)
	hainan_y = [random.randint(15, 25) for t in range(0, 24)]
	# 设置画板属性
	plt.figure(figsize = (10, 8), dpi = 100)
	# 往画板绘图
	plt.plot(hainan_x, hainan_y, label="海南")
	# 模拟北京一天内温度的变化
	# 生成x轴的24小时
	beijing_x = [h for h in range(0, 24)]
	# 生成y轴的温度随机值(5, 10)
	beijing_y = [random.randint(5, 10) for t in range(0, 24)]
	# 往画板绘图
	plt.plot(beijing_x, beijing_y, label="北京")
	# 模拟河北一天内温度的变化
	hebei_x = beijing_x
	hebei_y = [random.randint(1, 5) for t in range(0, 24)]
	# 自定义绘制属性: 颜色color="#0c8ac5", linestyle"-"""--""-.":", 线宽linewidth, 透明度alpha
	plt.plot(hebei_x, hebei_y, label="河北",color="#823384", linestyle=":", linewidth=3, alpha=0.3)
	# 坐标轴显示设置
	# 生成24小时的描述
	x_ = [x_ for x_ in range(0, 24)]
	x_desc = ["{}时".format(x_desc) for x_desc in x_]
	# 设置x轴显示 24小时
	plt.xticks(x_, x_desc)
	# 生成10至30度的描述
	y_ = [y_ for y_ in range(0, 30)][::2]
	y_desc = ["{}℃".format(y_desc) for y_desc in y_]
	# 设置y轴显示温度描述
	plt.yticks(y_, y_desc)
	# 指定x y轴的名称
	plt.xlabel("时间")
	plt.ylabel("温度")
	# 指定标题
	plt.title("一天内温度的变化")
	# 显示图例
	plt.legend(loc="best")
	# 将数据生成图片, 保存到当前目录下
	plt.savefig("./t.png")
	# 在浏览器内展示图片
	plt.show()

def drawBar():
	# 条形图绘制名侦探柯南主要角色年龄
	role_list = ["柯南", "毛利兰", "灰原哀", "琴酒","贝尔摩德", "伏特加", "赤井秀一", "目暮十三"]
	role_age = [7, 17, 7, 34, 32, 30, 27, 46]
	# 实际年龄
	role_ture_age = [18, 17, 18, 34, 45, 30, 27, 46]
	x = np.array([i for i in range(1, len(role_list)+1)])
	y = np.array(role_age)
	y2 =np.array(role_ture_age)
	# 设置画板属性
	plt.figure(figsize = (15, 8), dpi = 100)
	# width以x为基准,向右为正,向左为负(如果多了,就需要为基准x加减响应的数值)
	plt.bar(x-0.15, y, width = 0.3, label="现实年龄", color="#509839")
	plt.bar(x+0.15, y2, width = 0.3, label="实际年龄", color="#c03035")
	x_ = [i for i in range(0, len(role_list)+1)]
	x_desc = ["{}".format(x_desc) for x_desc in role_list]
	x_desc.insert(0, "")
	y_ = range(0, 50)[::5]
	y_desc = ["{}岁".format(y_desc) for y_desc in range(0, 50)][::5]
	# x轴的数值和描述
	plt.xticks(x_, x_desc)
	plt.yticks(y_, y_desc)
	plt.xlabel("角色姓名")
	plt.ylabel("年龄")
	plt.title("名侦探柯南主要角色年龄(部分)")
	plt.legend(loc="best")
	plt.savefig("./mzt.png")
	plt.show()

def drawHist():
	# 时长数据
	time = [131,  98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119, 128, 121, 142, 127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115,  99, 136, 126, 134,  95, 138, 117, 111,78, 132, 124, 113, 150, 110, 117,  86,  95, 144, 105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123,  86, 101,  99, 136,123, 117, 119, 105, 137, 123, 128, 125, 104, 109, 134, 125, 127,105, 120, 107, 129, 116, 108, 132, 103, 136, 118, 102, 120, 114,105, 115, 132, 145, 119, 121, 112, 139, 125, 138, 109, 132, 134,156, 106, 117, 127, 144, 139, 139, 119, 140,  83, 110, 102,123,107, 143, 115, 136, 118, 139, 123, 112, 118, 125, 109, 119, 133,112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135,115, 146, 137, 116, 103, 144,  83, 123, 111, 110, 111, 100, 154,136, 100, 118, 119, 133, 134, 106, 129, 126, 110, 111, 109, 141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126,114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 133, 137,  92,121, 112, 146,  97, 137, 105,  98, 117, 112,  81,  97, 139, 113,134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 101, 110,105, 129, 137, 112, 120, 113, 133, 112,  83,  94, 146, 133, 101,131, 116, 111,  84, 137, 115, 122, 106, 144, 109, 123, 116, 111,111, 133, 150]
	max_time = max(time)
	min_time = min(time)
	# 指定分组宽度
	width = 5
	# 指定分组数量
	num_bins = int((max_time - min_time)/2)
	# 直方图统计电影时长频数
	plt.figure(figsize=(15, 8), dpi=80)
	# 绘制直方图
	plt.hist(time, num_bins,edgecolor='#000000', color="#509839",normed=1)
	# 指定显示刻度的个数 
	x_ = [i for i in range(min_time, max_time+1)]
	plt.xticks(x_[::width])
	# 显示网格
	plt.grid(True, linestyle="--", alpha=0.5)
	# 指定标题
	plt.title("Top250的IMDB电影时长统计")
	plt.savefig("./IMDB.png")
	plt.show()

def drawPie():
	# 学习时间分配
	pro_name = ["C++", "Python", "Java", "Go", "Swift"]
	pro_time = [10, 15, 5, 3, 1]
	# 画饼
	plt.pie(pro_time, labels=pro_name, autopct="%3.2f%%", colors=["#ea6f5a", "#509839", "#0c8ac5", "#d29922", "#fdf6e3"])
	# 指定标题
	plt.title("学习时间分配")
	# 保证为图形为正圆
	plt.axis("equal")
	# 显示图示
	plt.legend(loc="best")
	plt.savefig("./pro_learn.png")
	plt.show()


def draw3D():
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(-5, 5, 0.1)
	Y = np.arange(-5, 5, 0.1)
	X, Y = np.meshgrid(X, Y)
	R = np.sqrt(X ** 2 + Y ** 2)
	Z = np.sin(R)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	# 画表面,x,y,z坐标， 横向步长，纵向步长，颜色，线宽，是否渐变
	ax.set_zlim(-1.01, 1.01)  # 坐标系的下边界和上边界
	ax.zaxis.set_major_locator(LinearLocator(10))  # 设置Z轴标度
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # Z轴精度
	fig.colorbar(surf, shrink=0.5, aspect=5)  # shrink颜色条伸缩比例（0-1），aspect颜色条宽度（反比例，数值越大宽度越窄）
	plt.savefig("./3d.png")
	plt.show()

if __name__=='__main__':
	drawDemo()
	# drawPoint()
	# drawLine()
	# drawBar()
	# drawHist()
	# drawPie()
	# draw3D()
