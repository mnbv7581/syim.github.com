'''
Created on 2018. 10. 30.

@author: seongyong-im
'''
import os
import cv2
import numpy as np
from random import shuffle

import tensorflow as tf

from Learnit.VGG.utils import *

from Learnit.VGG.Define import *
from Learnit.VGG.VGG16 import *

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

vgg = VGG16(input_var, training_flag)

learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
 
#loss
ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_var, logits = vgg))
 
#optimizer : loss값을 정의하기위해 사용 되는 것 (정답과 예측값이 최소화가 될수 있도록 
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9) #epsilon = 1e-4
train = optimizer.minimize(ce)
 
#accuracy 
# argmax : 가장큰 인덱스를 가져옴
correct_prediction = tf.equal(tf.argmax(vgg, 1), tf.argmax(label_var, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())
sess = tf.Session()

sess.run(tf.global_variables_initializer())
saver.restore(sess, './model/vgg16_300.ckpt')

list_input_var = []
list_label_var = []

for test_name in test_names:
    #print(train_dir + train_name)

    img = cv2.imread(test_dir + test_name)
    cls = int(test_name[0])

    tf_img = img / 255.
    label = one_hot(cls, CLASSES)

    list_input_var.append(tf_img.copy())
    list_label_var.append(label.copy())

    # batch size : 16
    if len(list_input_var) == 100:
        np_input_var = np.asarray(list_input_var, dtype = np.float32)
        np_label_var = np.asarray(list_label_var, dtype = np.float32)

        input_map = { input_var : np_input_var,
                label_var : np_label_var,
                training_flag : False }
        batch_acc = accuracy.eval(session=sess,feed_dict = input_map)
        outputs = sess.run(vgg, feed_dict = {input_var : np_input_var, training_flag : False})
       
        print('pred class : {}, ground truth class : {}'.format(np.argmax(outputs, 1), np.argmax(np_label_var, 1)))
        print (batch_acc)
        list_input_var = []
        list_label_var = []