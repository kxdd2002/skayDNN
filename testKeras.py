'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import add, Input, Dense, Activation, Dropout, Flatten, BatchNormalization, Concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Convolution2D, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
import keras.backend  as K

import numpy as np

# --------------------------------------------------   cfg init    --------------------------------------------------

batch_size = 128
epochs = 10
num_classes = 10

# --------------------------------------------------   data init    --------------------------------------------------

def MNIST():
	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(60000, 28,28,1)
	x_test = x_test.reshape(10000, 28,28,1)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	return x_train,y_train,x_test,y_test

def CIFAR10():
	(x_train,y_train),(x_test,y_test) = cifar10.load_data()
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test= keras.utils.to_categorical(y_test, num_classes)
	# train_datagen = getDataGenerator(train_phase=True)
	# train_datagen = train_datagen.flow(x_train,y_train,batch_size = batch_size)
	# validation_datagen = getDataGenerator(train_phase=False)
	# validation_datagen = validation_datagen.flow(x_test,y_test,batch_size = batch_size)
	return x_train,y_train,x_test,y_test

def getDataGenerator(train_phase,rescale=1./255):
	if train_phase == True:
		datagen = ImageDataGenerator(
		rotation_range=0.,
		width_shift_range=0.05,
		height_shift_range=0.05,
		shear_range=0.05,
		zoom_range=0.05,
		channel_shift_range=0.,
		fill_mode='nearest',
		horizontal_flip=True,
		vertical_flip=False,
		rescale=rescale)
	else: 
		datagen = ImageDataGenerator( rescale=rescale )
	return datagen

# --------------------------------------------------   model init    --------------------------------------------------

def FC2(num_classes=10,img_dim=(28,28,1)):
	model = Sequential()
	# relu sigmoid
	model.add(Flatten(input_shape=img_dim))
	model.add(Dense(50, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(100, activation='relu'))
	# model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	# model.summary()
	# model.compile(loss='categorical_crossentropy',
	#             optimizer=RMSprop(),
	#             metrics=['accuracy'])
	return model

##### ILSVRC https://www.cnblogs.com/skyfsm/p/8451834.html

# 1998 LeNet
def LeNet(nb_classes=num_classes, img_dim=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=img_dim,padding='valid',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	model.add(Dense(num_classes,activation='softmax'))
	return model

# 2012 AlexNet
def AlexNet(nb_classes=num_classes, img_dim=(224,224,3)):
	model = Sequential()
	model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=img_dim,padding='valid',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes,activation='softmax'))
	return model

# 2013
def ZF_Net(nb_classes=num_classes, img_dim=(224,224,3)):
	model = Sequential()  
	model.add(Conv2D(96,(7,7),strides=(2,2),input_shape=img_dim,padding='valid',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
	model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
	model.add(Flatten())  
	model.add(Dense(4096,activation='relu'))  
	model.add(Dropout(0.5))  
	model.add(Dense(4096,activation='relu'))  
	model.add(Dropout(0.5))  
	model.add(Dense(num_classes,activation='softmax'))  
	return model

# 2014 VGG16
def VGG_16(nb_classes=num_classes, img_dim=(224,224,3)):   
	model = Sequential()
	model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=img_dim,padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes,activation='softmax'))
	return model

# 2014 GoogLeNet
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
	x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
	x = BatchNormalization(axis=3,name=bn_name)(x)
	return x

def Inception(x,nb_filter):
	branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
	branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
	branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)
	branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
	branch5x5 = Conv2d_BN(branch5x5,nb_filter,(5,5), padding='same',strides=(1,1),name=None)
	branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
	branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)
	x = Concatenate(axis=3)([branch1x1,branch3x3,branch5x5,branchpool])
	return x

def GoogLeNet(nb_classes=num_classes, img_dim=(224,224,3)):
	inpt = Input(shape=img_dim)
	#padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
	x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
	x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
	x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
	x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
	x = Inception(x,64)#256
	x = Inception(x,120)#480
	x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
	x = Inception(x,128)#512
	x = Inception(x,128)
	x = Inception(x,128)
	x = Inception(x,132)#528
	x = Inception(x,208)#832
	x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
	x = Inception(x,208)
	x = Inception(x,256)#1024
	x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
	x = Dropout(0.4)(x)
	x = Dense(1000,activation='relu')(x)
	x = Dense(num_classes,activation='softmax')(x)
	model = Model(inputs=inpt,outputs=x,name='inception')
	return model

# 2015 ResNet50 何恺明
# https://blog.csdn.net/loveliuzz/article/details/79117397
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
	x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
	x = BatchNormalization(axis=3,name=bn_name)(x)
	return x

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')
	if with_conv_shortcut:
		shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)
		x = add([x,shortcut])
		return x
	else:
		x = add([x,inpt])
		return x

def ResNet50(nb_classes=num_classes, img_dim=(224,224,3)):
	inpt = Input(shape=img_dim)
	x = ZeroPadding2D((3,3))(inpt)
	x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
	x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
	x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)
	x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
	x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
	x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
	x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))
	x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))
	x = AveragePooling2D(pool_size=(7,7))(x)
	x = Flatten()(x)
	x = Dense(num_classes,activation='softmax')(x)
	model = Model(inputs=inpt,outputs=x)
	return model

# 2017 DenseNet121
"""
This DenseNet implementation comes from:
	https://github.com/titu1994/DenseNet/blob/master/densenet_fast.py
I've made some modifications so as to make it consistent with Keras2 interface
"""
def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
	x = Activation('relu')(input)
	x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
					  kernel_regularizer=l2(weight_decay))(x)
	if dropout_rate is not None:
		x = Dropout(dropout_rate)(x)
	return x

def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
	concat_axis = 1 if K.image_dim_ordering() == "th" else -1
	x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
					  kernel_regularizer=l2(weight_decay))(input)
	if dropout_rate is not None:
		x = Dropout(dropout_rate)(x)
	x = AveragePooling2D((2, 2), strides=(2, 2))(x)
	x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)
	return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
	concat_axis = 1 if K.image_dim_ordering() == "th" else -1
	feature_list = [x]
	for i in range(nb_layers):
		x = conv_block(x, growth_rate, dropout_rate, weight_decay)
		feature_list.append(x)
		x = Concatenate(axis=concat_axis)(feature_list)
		nb_filter += growth_rate
	return x, nb_filter

def DenseNet(nb_classes=num_classes, img_dim=(224,224,3), depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
					 weight_decay=1E-4, verbose=True):
	''' Build the create_dense_net model
	Args:
		nb_classes: number of classes
		img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
		depth: number or layers
		nb_dense_block: number of dense blocks to add to end
		growth_rate: number of filters to add
		nb_filter: number of filters
		dropout_rate: dropout rate
		weight_decay: weight decay
	Returns: keras tensor with nb_layers of conv_block appended
	'''
	model_input = Input(shape=img_dim)
	concat_axis = 1 if K.image_dim_ordering() == "th" else -1
	assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
	# layers in each dense block
	nb_layers = int((depth - 4) / 3)
	# Initial convolution
	x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False, kernel_regularizer=l2(weight_decay))(model_input)
	x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
	# Add dense blocks
	for block_idx in range(nb_dense_block - 1):
		x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
		# add transition_block
		x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)
	# The last dense_block does not have a transition_block
	x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
	x = Activation('relu')(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
	densenet = Model(inputs=model_input, outputs=x)
	if verbose: 
		print("DenseNet-%d-%d created." % (depth, growth_rate))
	return densenet

# --------------------------------------------------   train and opt model function    --------------------------------------------------

def trainMdoel(model, x_train, y_train, x_test, y_test,saveFile='model.h5'):
	history = model.fit(x_train, y_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=1,
						validation_data=(x_test, y_test))
	# 保存模型 model.save_weights("model.h5") / loaded_model.load_weights("model.h5") / model.load_weights('my_model_weights.h5', by_name=True)
	model.save(saveFile)   # HDF5文件，pip install h5py
	return history

def loadJson(saveFile='model.json'):
	# load json and create model
	json_file = open(saveFile, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	return loaded_model

def saveJson(model,saveFile='model.json'):
	# serialize model to JSON
	model_json = model.to_json()
	with open(saveFile, "w") as json_file:
		json_file.write(model_json)

def showWeights(saveFile='model.h5'):
	model = load_model(saveFile)
	# model.set_weights()
	weights = model.get_weights()
	for i in range(6):
		print(i, np.array(weights)[i].shape)
	# print('size:',len(weights[1]))
	# for i in range(10):
	#   print(i, np.array(weights)[1][i])

def evaluateModel(x_test, labels,saveFile='model.h5'):
	model = load_model(saveFile)
	score = model.evaluate(x_test, labels, verbose=0)
	return score

def drawMap(modle):
	pass

# --------------------------------------------------   train function    --------------------------------------------------

def testKeras():
	x_train,y_train,x_test,y_test = MNIST()
	model = FC2(img_dim=(28,28,1))
	model.summary()
	model.compile(loss='categorical_crossentropy',
				  optimizer=RMSprop(),
				  metrics=['accuracy'])
	his = trainMdoel(model,x_train,y_train,x_test,y_test)
	score = evaluateModel(x_test, y_test)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

# ---------------------------------------------------------  main funciton  ------------------------------------------------------

if __name__ == '__main__':
	testKeras()
	model = load_model('model.h5')
	saveJson(model)








