import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_img(img_path = 'test1.png'):
	src = np.array(Image.open(img_path))
	print(src.shape) 
	return src

def draw_img(src, p = []):
	plt.imshow(src)
	for pi in p:
		plt.plot(pi[0], pi[1], '.')
	plt.axis('image')
	plt.show()

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

def use_homo_warp(imgsrc, homo_mat, dst_size=(600,800)):
	if len(imgsrc.shape)==2:
		return use_homo( imgsrc.reshape(imgsrc.shape[0], imgsrc.shape[1] ,1), homo_mat, dst_size ).reshape( dst_size[0],dst_size[1] )
	return use_homo(imgsrc, homo_mat, dst_size)

# imgsrc (2,m*n) A (2*2) B(2*1)
def use_affine_warp(imgsrc, A, B, dst_size=(350,600)):
	homo = np.vstack((np.hstack((A,np.array(B).reshape(2,1))),(0,0,1)))
	return use_homo_warp(imgsrc,homo,dst_size)

def get_homo_mat(A=None,B=None):
	if A and B:
		return np.vstack((np.hstack((A,np.array(B).reshape(2,1))),(0,0,1)))
	homo1 = np.array( [[ 1.7815545e+00 , 1.8503717e-01 , -1.4120541e+02],[-2.0089908e-02 , 2.3985806e+00 , -6.4675806e+02],[ 8.9432091e-05 , 3.5880026e-04 , 1.0000000e+00]])
	return homo1

def testhomo():
	src = load_img('p1.jpg')
	homo = get_homo_mat()
	dst = use_homo_warp(src,homo,(600,800))
	Image.fromarray(np.uint8(dst)).save('warp.jpg')

testhomo()

