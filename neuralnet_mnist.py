# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, relu

def saveObj(obj,objFile):
    with open(objFile, 'wb') as f:
        pickle.dump(obj, f)

def loadObj(objFile):
    with open(objFile, 'rb') as f:
        return pickle.load(f)

def init_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network(model_file='sample_weight.pkl'):
    with open(model_file, 'rb') as f:
        network = pickle.load(f)
    print(network['W1'].shape)
    print(network['b1'].shape)
    print(network['W2'].shape)
    print(network['b2'].shape)
    print(network['W3'].shape)
    print(network['b3'].shape)
    print(type(network['W1'][0][0]))
    return network

def trans_int8(network):
    network['W1'] = network['W1'].astype(np.int8)
    network['b1'] = network['b1'].astype(np.int8)
    network['W2'] = network['W2'].astype(np.int8)
    network['b2'] = network['b2'].astype(np.int8)
    network['W3'] = network['W3'].astype(np.int8)
    network['b3'] = network['b3'].astype(np.int8)
    return network

def get_weights_from_h5(saveFile='model.h5'):
    from keras.models import load_model
    model = load_model(saveFile)
    # model.set_weights()
    weights = model.get_weights()
    for i in range(6):
        print(i, np.array(weights)[i].shape)
    return weights

def init_network_from_h5(weight_file):
    network = {}
    weights =  np.array(showWeights(weight_file))
    network['W1'] = weights[0]
    network['b1'] = weights[1]
    network['W2'] = weights[2]
    network['b2'] = weights[3]
    network['W3'] = weights[4]
    network['b3'] = weights[5]
    print(type(network['W1'][0][0]))
    return network

def predict(network, x, action):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = action(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = action(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

def test(network, action):
    x, t = init_data()
    batch_size = 100 # 批数量
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch, action)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

if __name__ == '__main__':
    print('-------------------test1-------------------')
    network = init_network()
    test(network,sigmoid)
    # print('-------------------test2-------------------')
    # network2 = init_network_from_h5('model2.h5')
    # test(network2,relu)
    # saveObj(network2,'keras_weight.pkl')
    # print('-------------------test3-------------------')
    # network3 = init_network_from_h5('model.h5')
    # test(network3,relu)
    print('-------------------test4-------------------')
    network4 = init_network('keras_weight.pkl')
    test(network4,relu)
    # network5 = trans_int8(network4)
    # saveObj(network5,'keras_weight_int8.pkl')
    print('-------------------test5-------------------')
    network5 = init_network('keras_weight_int8.pkl')
    test(network5,relu)

