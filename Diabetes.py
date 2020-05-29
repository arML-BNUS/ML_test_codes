# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:41:33 2020

@author: Asif
"""


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.txt', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[0:500,0:8]
y = dataset[0:500,8]

X_test = dataset[500:768,0:8]
y_test = dataset[500:768,8]

...
# define the keras model
model = Sequential()
model.add(Dense(40, input_dim=8, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss: %.2f' % (loss))
print('Accuracy: %.2f' % (accuracy*100))
