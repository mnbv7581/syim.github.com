import os
import glob

import cv2
import numpy as np
from random import shuffle

import tensorflow as tf

from Define import *
from ResNet import *

def normalize(vector):
    min_value = np.min(vector)
    max_value = np.max(vector)

    vector -= min_value
    vector /= (max_value - min_value)

    return vector

def heatmap_to_bbox(heatmap, threshold = 0.8):
    
    h, w = heatmap.shape
    xmin = ymin = h * 2
    xmax = ymax = 0

    heatmap_threshold = threshold * 255
    
    for y in range(h):
        for x in range(w):
            if heatmap[y, x] > heatmap_threshold:
                xmin = min(xmin, x)
                ymin = min(ymin, y)
                xmax = max(xmax, x)
                ymax = max(ymax, y)

    return [xmin, ymin, xmax, ymax]

if __name__ == '__main__':

    #train & test db load
    train_dir = '../../DB/image/test/'

    train_names = os.listdir(train_dir)
    shuffle(train_names)
    
    #path define
    model_path = './model/'
    model_name = 'resnet_{}.ckpt'
    
    input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input')
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

            _bbox_heatmap = _heatmap.copy()
            
            _heatmap = cv2.applyColorMap(_heatmap, cv2.COLORMAP_JET)
            _heatmap = cv2.resize(_heatmap, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

            _bbox_heatmap = cv2.resize(_bbox_heatmap, (224, 224), interpolation = cv2.INTER_CUBIC)
            xmin, ymin, xmax, ymax = heatmap_to_bbox(_bbox_heatmap)

            img = cv2.addWeighted(img, 0.5, _heatmap, 0.5, 0)
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            cv2.imshow('show', img)
            cv2.waitKey(0)
