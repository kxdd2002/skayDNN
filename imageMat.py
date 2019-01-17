import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_img(img_path = 'test1.png'):
	src = np.array(Image.open(img_path))
		# .convert('L'))
	# src=src.convert('L')
	print(src.shape) 
	return src

def draw_img(src, p = None):
	mpl.rcParams.update({'image.cmap': 'gray',
						 'lines.markersize': 12,
						 'axes.prop_cycle': mpl.cycler('color', ['#00ff00'])})
	plt.imshow(src)
	for pi in p:
		plt.plot(pi[0], pi[1], '.')
	plt.axis('image')
	plt.show()

# imgsrc (2,m*n) A (2*2) B(2*1)
def affine(imgsrc, A, B, dst_size=(350,600)):
	H, W, C = imgsrc.shape if len(imgsrc.shape)==3 else (*imgsrc.shape,1)
	points = np.mgrid[0:W, 0:H].reshape((2, H*W)).transpose(1,0)
	new_points = np.dot(points, A)+B
	x, y = new_points.transpose(1,0).reshape((2,H,W),order='F').round().astype(int)
	remap_index = (x + W*y)[:dst_size[0],:dst_size[1]]
	# print(imgsrc.shape)
	# print(remap_index.shape)
	dst = []
	if C == 1:
		dst = np.take(imgsrc, remap_index, mode='wrap')
		return dst
	for c in range(C):
		dstc = np.take(imgsrc[:,:,c], remap_index, mode='wrap')
		# print(c, dstc.shape)
		dst.append(dstc)
	dst = np.array(dst).transpose(1,2,0)
	# print(dst.shape)
	return dst

def get_affine_mat(p,dst_size):
	# print(dst_size)
	w,h = dst_size if len(dst_size)==2 else dst_size[:2]
	# dst_points = [(0,0),(dw,0),(dw,dh),(0,dh)]
	# A = np.array([[(p[1][0]-p[0][0])/w, (p[2][0]-p[0][0])/h], [(p[1][1]-p[0][1])/w, (p[2][1]-p[0][1])/h]])
	# B = p[0]
	# A = [[-1.2,0], [0,-1.2]]
	# B = [0, -150]
	# A = [[1.08585758, 0.07642399], [0.96166859, 2.41372448]]
	# B = [201, 102]
	A = [[ 1.7815545e+00 , 1.8503717e-01], [-2.0089908e-02 , 2.3985806e+00]]
	B = [ -1.4120541e+02 ,  -6.4675806e+02]
	temp = np.linalg.inv(np.vstack((np.hstack((A,np.array(B).reshape(2,1))),(0,0,1))))
	A = temp[:2,:2]
	B = temp[:2,2].reshape((2,))
	print('get_affine_mat', A, B)
	return A,B

def get_homo_mat(A=None,B=None):
	if A and B:
		return np.vstack((np.hstack((A,np.array(B).reshape(2,1))),(0,0,1)))
	homo1 = np.array( [[ 1.7815545e+00 , 1.8503717e-01 , -1.4120541e+02],[-2.0089908e-02 , 2.3985806e+00 , -6.4675806e+02],[ 8.9432091e-05 , 3.5880026e-04 , 1.0000000e+00]])
	return homo1

def use_homo_warp(imgsrc, homo_mat, dst_size=(600,800)):
	if len(imgsrc.shape)==2:
		return use_homo( imgsrc.reshape(imgsrc.shape[0], imgsrc.shape[1] ,1), homo_mat, dst_size ).reshape( dst_size[0],dst_size[1] )
	return use_homo(imgsrc, homo_mat, dst_size)

def use_homo(imgsrc, homo_mat, dst_size=(600,800)):
	H, W, C = imgsrc.shape
	if dst_size[0]>H:
		imgsrc = np.vstack(( imgsrc, np.zeros(( dst_size[0]-H , W , C )) ))
		H = dst_size[0]
	if dst_size[1]>W:
		imgsrc = np.hstack(( imgsrc, np.zeros(( H , dst_size[1]-W , C )) ))
		W = dst_size[1]
	points = np.vstack(( np.mgrid[0:W,0:H].reshape(( 2,W*H )) , np.ones(( 1,W*H )) )) # pos index is (:,px*H+py)
	new_points = np.dot(np.linalg.inv(homo_mat), points) 
	x, y, z = new_points.reshape((3,H,W),order='F')  # pos index is (py,px)
	x, y = (x/z).round().astype(int),(y/z).round().astype(int)
	dst = np.array( [ np.take(imgsrc[:,:,c], (x + W*y)) for c in range(C)] ).transpose(1,2,0)[:dst_size[0],:dst_size[1]] # mode='wrap'
	return dst

import cv2

def test():
	src = load_img('p1.jpg')
	# print(src[0,0])
	# p = [(201, 102), (580, 78), (790, 412), (50,443)]
	# draw_img(src, p)
	A,B = get_affine_mat(None,src.shape)
	dst = affine(src, A, B)
	print('dst',dst.shape)
	# draw_img(dst)
	cv2.imwrite('pp1.jpg',dst[:,:,[2,1,0]])

def test3d():
	src = load_img('p1.jpg')
	# print(src[0,0])
	# p = [(50, 280), (580, 78), (790, 412), (50,443)]
	p = (385,432)
	# draw_img(src)
	W,H = 800,600
	rw,rh = src.shape[1]/W,src.shape[0]/H
	# print(rw,rh)
	homo = get_homo_mat()
	# homo = np.dot(np.array([rw, 0.0, 0.0, 0.0, rh, 0.0, 0.0, 0.0, 1.0],np.float32).reshape(3,3),homo)
	dst = use_homo_warp(src,homo)
	print('dst',dst.shape)
	cv2.imwrite('pp.jpg',dst[:,:,[2,1,0]])
	# draw_img(dst)

def testcv():
	src = load_img('pp3.jpg')
	p = [(385,432)]
	
	# homo = get_homo_mat()
	# dst = cv2.warpPerspective(src,homo,(800,600))
	# finalp = [(530,315)]
	finalp = [(525,320)]
	draw_img(src,finalp)
	# cv2.imwrite('pp3.jpg',dst[:,:,[2,1,0]])

test3d()

