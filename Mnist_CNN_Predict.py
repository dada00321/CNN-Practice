# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:15:48 2019

@author: 88696
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:02:47 2019

@author: 88696
"""
import glob, cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def show_predictions(images,labels,predictions,start_id,num=10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        #顯示黑白圖片
        ax.imshow(images[start_id], cmap='binary')
        
        # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(predictions) > 0 ) :
            title = 'ai = ' + str(predictions[i])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[i]==labels[i] else ' (x)') 
            title += '\nlabel = ' + str(labels[i])
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[i])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()

#建立測試樣本(features & labels)
files = glob.glob("images\*.jpg")
test_feature = []
test_label = []
for file in files:
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) #灰階(gray)
    _, img = cv2.threshold(img, 120, 255,
          cv2.THRESH_BINARY_INV) #轉為反向黑白
    test_feature.append(img)
    #label = file[7]
    label = file[list(file).index("s")+2] # ex: 3.jpg 的 "3"
    test_label.append(int(label))
    
test_feature = np.array(test_feature)
test_label = np.array(test_label)

test_feature_vector = test_feature.reshape(len(test_feature),28,28,1).astype('float32')
test_feature_normalize = test_feature_vector/255

# laod model
print("load model Mnist_CNN_model.h5")
model = load_model('Mnist_CNN_model.h5')

# predict
my_pred = model.predict_classes(test_feature_normalize)

#show
show_predictions(images = test_feature, labels = test_label,
                 predictions = my_pred, start_id = 0, num = 10)