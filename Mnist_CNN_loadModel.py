# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:54:43 2019

@author: 88696
"""

import matplotlib.pyplot as plt

def show_predictions(images, labels, predictions, start_id, num):
    plt.gcf().set_size_inches(12,14)
    if num>25:  num = 25    
    for i in range(num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(images[start_id], cmap='binary')      
        if(len(predictions) > 0):
            title = 'ai = ' + str(predictions[i])
            title += ("(O)" if predictions[i] == labels[i] else "(X)")
            title += "\nlabel = " + str(labels[i])
        else:
            title = "label = " + str(labels[i])      
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        start_id += 1
    plt.show()
    
'''
 1. Pre-process
'''
from keras.datasets import mnist
from keras.utils import np_utils
# load data
(train_feature, train_label), \
(test_feature, test_label) = mnist.load_data()

# feature scaling
train_feature_vector = train_feature.reshape(len(train_feature),train_feature.shape[1],train_feature.shape[2],1).astype('float32')  #將feature轉為(60000,28,28,1)的四維向量,並轉型為float標準化  
test_feature_vector = test_feature.reshape(len(test_feature),test_feature.shape[1],test_feature.shape[2],1).astype('float32')   #將feature轉為(10000,28,28,1)的四維向量,並轉型為float標準化  

# feature normalization
train_feature_normalized = train_feature_vector/255
test_feature_normalized = test_feature_vector/255

# convert typr of label to One-Hot Encoding
train_label_one_hot = np_utils.to_categorical(train_label)
test_label_one_hot = np_utils.to_categorical(test_label)

'''
 2. Loading model
'''
from keras.models import load_model
print("loading Mnist_CNN_model.h5 ...")
model = load_model('Mnist_CNN_model.h5')

'''
 3. Evaluate the accuracy of this trained model
'''
accuracy = model.evaluate(test_feature_normalized,test_label_one_hot)
print("\n","accuracy:",accuracy[1])

'''
 4. Making predictions
'''
pred = model.predict_classes(test_feature_normalized)
show_predictions(test_feature, test_label, pred, start_id=0, num=10)
