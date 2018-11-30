# coding=utf-8
import os, sys
import requests
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------感知器及模拟加法器---------------------------------------------------------

def Perceptor(x,a,b):
	y = np.dot(np.array(x),np.array(a)) + b
	return int(y > 0)

def AND(x):
	return Perceptor(x,[0.5,0.5],-0.6)

def OR(x):
	return Perceptor(x,[0.5,0.5],-0.4)

def NAND(x):
	return Perceptor(x,[-0.5,-0.5],0.6)

def XOR(x):
	return AND([NAND(x),OR(x)])

def testFunc(func):
	xs = np.array([[0,0],[0,1],[1,0],[1,1]])
	for x in xs:
		print(x,'==>',func(x))

def test1():
	print('AND test:')
	testFunc(AND)
	print('OR test:')
	testFunc(OR)
	print('NAND test:')
	testFunc(NAND)
	print('XOR test:')
	testFunc(XOR)

# ----------------------------------------------------------------------------神经网络---------------------------------------------------------

def initModel():
	data={}
	return data

test1()