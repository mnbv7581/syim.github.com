import os
import cv2
import numpy as np
from random import shuffle

import tensorflow as tf

from Learnit.DenseNet.utils import *

from Learnit.DenseNet.Define import *
from Learnit.DenseNet.DenseNet import *

if __name__ == '__main__':

    #train & test db load
    train_dir = '../../DB/image/train/'
    test_dir  = '../../DB/image/test/'

    train_names = os.listdir(train_dir)
    shuffle(train_names)

    valid_names = train_names[:5000]
    train_names = train_names[5000:]

    test_names  = os.listdir(test_dir)
    
    #path define
    model_path = './model/'
    model_name = 'densenet_{}.ckpt'
    
    #model build
    input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input')
    label_var = tf.placeholder(tf.float32, shape=[None, CLASSES])

    training_flag = tf.placeholder(tf.bool)
    densenet = DenseNet(input_var, training_flag)

    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    
    #loss
    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_var, logits = densenet))

    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-4)
    train = optimizer.minimize(ce)

    #accuracy
    correct_prediction = tf.equal(tf.argmax(densenet, 1), tf.argmax(label_var, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #save
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, model_path + model_name.format(99))

        #epoch
        print('set batch size :', BATCH_SIZE)

        #epoch_learning_rate = 1e-1 #momentum
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
                
                tf_img = img / 255.0
                label = one_hot(cls, CLASSES)

                list_input_var.append(tf_img.copy())
                list_label_var.append(label.copy())

                if len(list_input_var) == BATCH_SIZE:
                    np_input_var = np.asarray(list_input_var, dtype = np.float32)
                    np_label_var = np.asarray(list_label_var, dtype = np.float32)

                    input_map = { input_var : np_input_var,
                                  label_var : np_label_var,
                                  learning_rate : epoch_learning_rate,
                                  training_flag : True }
                    
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

            #validation
            if epoch % 5 == 0:
                list_input_var = []
                list_label_var = []

                _accuracy = 0.0
                _accuracy_sample_cnt = 0
                    
                for valid_name in valid_names:
                    img = cv2.imread(train_dir + valid_name)
                    cls = int(valid_name[0])

                    if img is None:
                        continue

                    tf_img = img / 255.0
                    label = one_hot(cls, CLASSES)

                    list_input_var.append(tf_img.copy())
                    list_label_var.append(label.copy())

                    if len(list_input_var) == BATCH_SIZE:
                        np_input_var = np.asarray(list_input_var, dtype = np.float32)
                        np_label_var = np.asarray(list_label_var, dtype = np.float32)

                        input_map = { input_var : np_input_var,
                                      label_var : np_label_var,
                                      training_flag : False }

                        batch_acc = accuracy.eval(feed_dict = input_map)
                        
                        _accuracy  += batch_acc
                        _accuracy_sample_cnt += 1
                        
                        list_input_var = []
                        list_label_var = []

                print('epoch {} valid set accuracy :'.format(epoch), _accuracy / _accuracy_sample_cnt)

                #test
                list_input_var = []
                list_label_var = []

                _accuracy = 0.0
                _accuracy_sample_cnt = 0

                for test_name in test_names:
                    img = cv2.imread(test_dir + test_name)
                    cls = int(test_name[0])

                    if img is None:
                        continue

                    tf_img = img / 255.0
                    label = one_hot(cls, CLASSES)

                    list_input_var.append(tf_img.copy())
                    list_label_var.append(label.copy())

                    if len(list_input_var) == BATCH_SIZE:
                        np_input_var = np.asarray(list_input_var, dtype = np.float32)
                        np_label_var = np.asarray(list_label_var, dtype = np.float32)

                        input_map = { input_var : np_input_var,
                                label_var : np_label_var,
                                training_flag : False }
                
                        batch_acc = accuracy.eval(feed_dict = input_map)
                        
                        _accuracy += batch_acc
                        _accuracy_sample_cnt += 1
                        
                        list_input_var = []
                        list_label_var = []
                
                print('epoch {} test set accuracy :'.format(epoch), _accuracy / _accuracy_sample_cnt)