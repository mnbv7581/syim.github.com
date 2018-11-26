'''
Created on 2018. 10. 28.

@author: seongyong-im
'''
import os
import cv2
import numpy as np
import tensorflow as tf

from random import shuffle
from Learnit.DenseNet.utils import *
from Learnit.DenseNet.Define import *
from Learnit.DenseNet.DenseNet import *

#model build
input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input') 
label_var = tf.placeholder(tf.float32, shape=[None, CLASSES])

training_flag = tf.placeholder(tf.bool)#dropout 사용 유무 때 사용 (traning떄 true 사용)   
vgg = DenseNet(input_var, training_flag)

#===============================================================================
# learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
# 
# #loss
# ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_var, logits = vgg))
# 
# #optimizer : loss값을 정의하기위해 사용 되는 것 (정답과 예측값이 최소화가 될수 있도록 
# optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9) #epsilon = 1e-4
# train = optimizer.minimize(ce)
# 
# #accuracy 
# # argmax : 가장큰 인덱스를 가져옴
# correct_prediction = tf.equal(tf.argmax(vgg, 1), tf.argmax(label_var, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#===============================================================================

print(vgg)
Image_name = '9_articulated_lorry_s_000692.png'
Image_name_pars = Image_name.split('_')
img = cv2.imread('./test/9_articulated_lorry_s_000692.png')
list_input_var = []
list_label_var = []
cls = int(Image_name_pars[0])
label = one_hot(cls, CLASSES)

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
saver.restore(sess, './model/densenet_70.ckpt')
tf_img = img / 255.0

list_input_var.append(tf_img.copy())
list_label_var.append(label.copy())

np_input_var = np.asarray(list_input_var, dtype = np.float32)
np_label_var = np.asarray(list_label_var, dtype = np.float32)

output = sess.run(vgg, feed_dict = {input_var : np_input_var, training_flag : False})

print('pred class : {}, ground truth class : {}'.format(np.argmax(output, 1), np.argmax(np_label_var, 1)))
 
    
    