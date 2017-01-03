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
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

spn = SPN()
spn.make_random_model(((30, 60), (2, 5)), 784, 10, cont=True, classify=True)#step=step)
# my_spn.continuous = True
spn.start_session()

spn.train(1, data=X_train, labels=Y_train, count=False)

#accuracy:
total_loss = 0.0
total_right = 0
total = 0
ms = 512
a = 0
b = 0
for i in range(1+(len(X_test)-1)//ms):
	print i + 1, '/', 1+(len(X_test)-1)//ms
	b = min((i+1)*ms, len(X_test))
	if a == b:
		break;
	test_loss = spn.evaluate(X_test[a:b], Y_test[a:b])[0]
	total_loss += test_loss*(float(b - i*ms))
    total += (b-i*ms)
    pred = spn.predict(X_test[a:b])
    argz = np.argmax(pred, axis=1)
    argz2 = np.argmax(Y_test[a:b])
    f = np.sum(np.array(argz==argz2))
    total_right += f
    test_loss = total_loss/float(len(my_spn.data.test))
    total_loss = 0.0
print total_right, '/', total, ' or ', str(float(total_right)/float(total)) + '%'

