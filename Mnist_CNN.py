# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:47:15 2019

@author: 88696
"""
'''
def show_image(image):
    fig = plt.gcf() # get current figure
    fig.set_size_inches(2,2) 
    plt.imshow(image, cmap='binary') # 黑白灰階顯示
    plt.show()
'''

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
 2. Build CNN model
'''
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# build basic model
model = Sequential()

# build Convolution Layer - 1
# get 10 of 28*28 pics
model.add(Conv2D(filters=10,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=(train_feature.shape[1],train_feature.shape[2],1), #(28,28,1)
                 activation='relu'))

# build Pooling Layer - 1
# turns out same number of 14*14 pics
model.add(MaxPooling2D(pool_size=(2,2)))

# build Convolution Layer - 2
# get 20 of 14*14 pics
model.add(Conv2D(filters=20,
                 kernel_size=(3,3),
                 padding='same',
                 #input_shape=(train_feature.shape[1],train_feature.shape[2],1), #(28,28,1)
                 activation='relu'))

# build Pooling Layer - 2
# turns out same number(20) of 7*7 pics
model.add(MaxPooling2D(pool_size=(2,2)))

# build Dropout Layer
model.add(Dropout(0.2)) # dropout 20 percent of data to prevent Overfitting

# build Flattrn Layer
# Adding this layer to transfer 20 of 7*7 pics from Pooling Layer 
# to a one-dimensional vector has 980(=20*7*7) units of nervous
model.add(Flatten())

# build Hidden Layer
model.add(Dense(units=256, activation='relu'))

# build Output Layer
model.add(Dense(units=10, activation='softmax'))

'''
 3. Train the model
'''
# define/set the training method
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# training 
train_hist = model.fit(x=train_feature_normalized,
                       y=train_label_one_hot,
                       validation_split=0.2,
                       epochs=10,
                       batch_size=200,
                       verbose=2)
'''
 4. Evaluate the accuracy of this trained model
'''
accuracy = model.evaluate(test_feature_normalized,test_label_one_hot)
print("\n","accuracy:",accuracy[1])

'''
 5. Making predictions
'''
pred = model.predict_classes(test_feature_normalized)
show_predictions(test_feature, test_label, pred, start_id=0, num=10)
