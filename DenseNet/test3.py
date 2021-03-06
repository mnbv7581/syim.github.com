'''
Created on 2018. 10. 30.

@author: seongyong-im
'''
import os
import cv2
import numpy as np
from random import shuffle

import tensorflow as tf

from Learnit.DenseNet.utils import *
from Learnit.DenseNet.Define import *
from Learnit.DenseNet.DenseNet import *

#train & test db load
train_dir = '../../DB/image/train/'
test_dir  = '../../DB/image/test/'

train_names = os.listdir(train_dir)


valid_names = train_names[:5000]
train_names = train_names[5000:]

test_names  = os.listdir(test_dir)
shuffle(test_names)
print('test set :', len(test_names))

input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input')
label_var = tf.placeholder(tf.float32, shape=[None, CLASSES])
training_flag = tf.placeholder(tf.bool)

vgg = DenseNet(input_var, training_flag)
 
learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
  
#loss
ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_var, logits = vgg))
  
#optimizer : loss값을 정의하기위해 사용 되는 것 (정답과 예측값이 최소화가 될수 있도록 
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-4)
train = optimizer.minimize(ce)
  
#accuracy 
# argmax : 가장큰 인덱스를 가져옴
correct_prediction = tf.equal(tf.argmax(vgg, 1), tf.argmax(label_var, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
saver = tf.train.Saver(tf.global_variables())
sess = tf.Session()
 
sess.run(tf.global_variables_initializer())
saver.restore(sess, './model/densenet_70.ckpt')
 
list_input_var = []
list_label_var = []

cap = cv2.VideoCapture(0)


while (cap.isOpened()):
    #print('width: {}, height : {}'.format(cap.get(3)), format(cap.get(4)))
    ret, fram = cap.read()
    
    if ret:  
        cv2.imshow('video', fram)    
        fram = cv2.resize(fram,(IMAGE_WIDTH, IMAGE_HEIGHT))
        tf_img = fram / 255.
        
        list_input_var.append(tf_img.copy())
       
        np_input_var = np.asarray(list_input_var, dtype = np.float32)
        
        output = sess.run(vgg, feed_dict = {input_var : np_input_var, training_flag : False})
         
        print('pred class : {}'.format(np.argmax(output, 1)))
        list_input_var = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('error')

cap.release()
cv2.destroyAllWindows()    
    

