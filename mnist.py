import time
from model import Model 
from SPN2 import SPN
import numpy as np
import random
import sys
from keras.datasets import mnist

nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

X_train = X_train.reshape(60000, 784, 1)
X_test = X_test.reshape(10000, 784, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 250
X_test /= 250
#X_train = np.round(X_train)
#X_test = np.round(X_test)
#X_train = np.array([[[0], [1]],[[1], [1]], [[1], [0]]]*2000)
#X_test = np.array([[[0], [1]], [[1],[1]], [[1], [0]]]*1000)
#X_train = np.concatenate([X_train, 1-X_train], axis=2)
#X_test = np.concatenate([X_test, 1-X_test], axis=2)
print X_test[0:10].tolist()
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)
#y_train = [0, 0, 1]*2000
#y_test = [0, 0, 1]*1000
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_train = 0.0*(Y_train - 1)  + Y_train

spn = SPN()
spn.make_random_model(((790, 800), (149, 150)), 784, 10, cont=True, classify=True)#step=step
# my_spn.continuous = True
spn.start_session()

spn.train(2, minibatch_size=1000, data=X_train, labels=Y_train, count=True, gd=True)
#spn.train(1, minibatch_size=1000, data=X_train, labels=Y_train, count=False, gd=True)

#accuracy:
total_loss = 0.0
total_right = 0
total = 0
ms = 1000
a = 0
b = 0
for i in range(1+(len(X_test)-1)//ms):
    print i + 1, '/', 1+(len(X_test)-1)//ms
    a = i*ms
    b = min((i+1)*ms, len(X_test))
    if a == b:
	break;
    test_loss = 0#spn.evaluate(X_test[a:b], Y_test[a:b])[0]
    total_loss += test_loss*(float(b - i*ms))
    total += (b-i*ms)
    pred = spn.predict(X_test[a:b])
    argz = np.argmax(pred, axis=1)
    argz2 = np.argmax(Y_test[a:b], axis=1)
    print zip(argz, argz2)
    f = np.sum(np.array(argz==argz2))
    total_right += f
    test_loss = total_loss/float(len(Y_test))
    total_loss = 0.0
print total_right, '/', total, ' or ', str(float(total_right)/float(total)) + '%'

