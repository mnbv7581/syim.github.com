#Weakly

import os
import glob

import cv2
import numpy as np
from random import shuffle

import tensorflow as tf

from Learnit.CAM.Define import *
from Learnit.CAM.ResNet import *

def one_hot(cls_idx, CLASSES):
    vector = np.zeros(CLASSES, dtype = np.float32)
    vector[cls_idx] = 1.
    return vector

if __name__ == '__main__':

    #train & test db load
    train_dir = '../../DB/image/train/'
    test_dir  = '../../DB/image/test/'
    
    train_names = os.listdir(train_dir)
    shuffle(train_names)

    #path define
    model_path = './model/'
    model_name = 'resnet_{}.ckpt'
    
    #model build
    input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input')
    label_var = tf.placeholder(tf.float32, shape=[None, CLASSES])

    training_flag = True
    net, conv, fc_w = ResNet(input_var, training_flag)

    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    
    #loss
    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_var, logits = net))

    #optimizer
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #학습때 피쳐들의 평균, 분산, 노멀라이즈 값들의 평균을 저장하여 사용하도록 함 
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-4)
        train = optimizer.minimize(ce)

    #accuracy
    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(label_var, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #save
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, model_path + model_name.format(148))
        #saver.save(sess, model_path + model_name.format(148))

        #epoch
        print('set batch size :', BATCH_SIZE)

        epoch_learning_rate = 1e-4
        for epoch in range(1, MAX_EPOCHS+1):
            if epoch == (MAX_EPOCHS * 0.5) or epoch == (MAX_EPOCHS * 0.75):
                epoch_learning_rate /= 10
            
            #init
            shuffle(train_names)
            list_input_var = []
            list_label_var = []

            train_cnt = 0
            train_acc = 0.0
            train_loss = 0.0

            #train
            for train_name in train_names:
                img = cv2.imread(train_dir + train_name)
                cls = int(train_name[0])

                if img is None:
                    continue
                
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                tf_img = img.astype(np.float32)
                label = one_hot(cls, CLASSES)

                list_input_var.append(tf_img.copy())
                list_label_var.append(label.copy())

                if len(list_input_var) == BATCH_SIZE:
                    np_input_var = np.asarray(list_input_var, dtype = np.float32)
                    np_label_var = np.asarray(list_label_var, dtype = np.float32)

                    input_map = { input_var : np_input_var,
                                  label_var : np_label_var,
                                  learning_rate : epoch_learning_rate }
                    
                    _, batch_loss = sess.run([train, ce], feed_dict = input_map)
                    batch_acc = accuracy.eval(feed_dict = input_map)

                    train_loss += batch_loss
                    train_acc  += batch_acc
                    train_cnt  += 1
                    
                    list_input_var = []
                    list_label_var = []

            #log
            print('epoch : {}, loss : {}, accuracy : {}'.format(epoch, train_loss / train_cnt, train_acc / train_cnt))

            #save
            saver.save(sess, model_path + model_name.format(epoch))



