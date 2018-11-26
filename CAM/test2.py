import os
import glob

import cv2
import numpy as np
from random import shuffle

import tensorflow as tf

from Learnit.CAM.Define import *
from Learnit.CAM.ResNet import *

# random -> 0~ 1
def normalize(vector):
    min_value = np.min(vector)
    max_value = np.max(vector)

    vector -= min_value
    vector /= (max_value - min_value)

    return vector

if __name__ == '__main__':

    #train & test db load
    train_dir = '../../DB/image/train/'
    test_dir  = '../../DB/image/test/'

    train_names = os.listdir(train_dir)
    
    #path define
    model_path = './model/'
    model_name = 'resnet_{}.ckpt'
    
    input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input')

    #test
    cls_idx = tf.placeholder(tf.int32)

    training_flag = False
    net, conv, fc_w = ResNet(input_var, training_flag)
    heatmaps = Visualize(cls_idx, conv, fc_w)
    
    #save
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path + model_name.format(300))

        for train_name in train_names:
            img = cv2.imread(train_dir + train_name)

            output = sess.run(net, feed_dict={input_var : [img]})[0]
            print(train_dir + train_name, ' : ', np.argmax(output))

            _heatmaps = sess.run(heatmaps, feed_dict={input_var :[img], cls_idx:np.argmax(output)})
            _heatmap = _heatmaps[0]

            norm_heatmap = normalize(_heatmap)

            _heatmap = norm_heatmap * 255.
            _heatmap = _heatmap.astype(np.uint8)
            _heatmap = cv2.applyColorMap(_heatmap, cv2.COLORMAP_JET)
            _heatmap = cv2.resize(_heatmap, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

            ori_img = img.copy()
            img = cv2.addWeighted(img, 0.5, _heatmap, 0.5, 0)
            img = cv2.resize(img, (112, 112), interpolation = cv2.INTER_CUBIC)

            cv2.imshow('show', img)
            cv2.imshow('heatmap', _heatmap)
            cv2.imshow('original', ori_img)
            cv2.waitKey(0)