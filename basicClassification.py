#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:53:10 2018

@author: KushDani
"""

#Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt #does not work

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
# training set                 test set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#labels are stored using array values ranging from 0 to 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#inspect first image in training set and see pixels fall between 0 and 255
train_images.shape
len(train_labels)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

#scale values to range between 0 and 1 and cast to float
train_images = train_images / 255.0
test_images = test_images / 255.0

#Display the first 25 images from the training set and display the class name below each image
#plt.figure(figsize=(10,10))
#for i in range(25):
    #plt.subplot(5,5,i+1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(train_images[i], cmap=plt.cm.binary) #actually displays training set
    #plt.xlabel(class_names[train_labels[i]])
    
#Chaining together simple layers ==> (basic building block of neural networks)
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), #transforms array from 2D to 1D of 784 pixels (28x28)
        #densely-connnected neural layers
        keras.layers.Dense(128, activation=tf.nn.relu), #has 128 nodes (neurons)
        #returns an array of 10 probability scores that sum to 1
        #each node contains probability score that the current image belongs to one of the 10 classes
        keras.layers.Dense(10, activation=tf.nn.softmax)])#10 node layer
    
#Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), #how the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy', #measures accuracy of model during training
              metrics=['accuracy']) #used to monitor the training and testing steps


#Train the neural network model; "fit" the model to the training "data"
model.fit(train_images, train_labels,epochs=5)

#test accuracy of test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)

print ('Test accuracy: ', test_acc)

predictions = model.predict(test_images) #model predicts label for each piece of data in training set
print('Prediction first: ', predictions[0]) #displays array of confidence levels based on each fashion article
print('Max confidence: ', np.argmax(predictions[0])) #displays max confidence fashion article
print('Actual first: ', test_labels[0]) #checks label of first element
