import os
import cv2
import numpy as np
from random import shuffle

import tensorflow as tf

from Learnit.VGG.utils import *

from Learnit.VGG.Define import *
from Learnit.VGG.VGG16 import *

if __name__ == '__main__':

    #train & test db load
    train_dir = '../../DB/image/train/'
    test_dir  = '../../DB/image/test/'

    train_names = os.listdir(train_dir)
    shuffle(train_names)

    valid_names = train_names[:5000]
    train_names = train_names[5000:]

    test_names  = os.listdir(test_dir)
    print('test set :', len(test_names))
    
    #path define
    model_path = './model/'
    model_name = 'vgg16_{}.ckpt'
    
    #model build
    input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input') 
    #학습을 하기위한 그래프(차원공간)을 미리 정의
    
    label_var = tf.placeholder(tf.float32, shape=[None, CLASSES])

    training_flag = tf.placeholder(tf.bool)#dropout 사용 유무 때 사용 (traning떄 true 사용)
    
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

    #save
    saver = tf.train.Saver(tf.global_variables())

    #session 이란 : 차원 그래프(공간)을 session에 올린다, 그리고 설정에따라 초기화, CPU/GPU 사용 등을 정의하여 실행할 수 있음)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #epoch
        print('set batch size :', BATCH_SIZE)
        
        epoch_learning_rate = 1e-2
        for epoch in range(1, MAX_EPOCHS+1):
            if epoch == (MAX_EPOCHS * 0.5) or epoch == (MAX_EPOCHS * 0.75): #Learning rate를 줄여가면서 학습을 시도
                epoch_learning_rate /= 10
            
            #init
            shuffle(train_names) #똑같은 순서대로 반복적인 학습을 하는 것 보다 여러 가지 상황(순서로) 데이터 학습을 시켜 일반적인 파라미터를 뽑는 것을 도와준다
            list_input_var = []
            list_label_var = []

            train_cnt = 0
            train_acc = 0.0
            train_loss = 0.0

            #train
            for train_name in train_names:
                img = cv2.imread(train_dir + train_name)
                cls = int(train_name[0]) #이미지 이름에 라벨링이 되어잇음 (파일확인해보면됨)

                if img is None:
                    continue
                
                tf_img = img / 255.0
                label = one_hot(cls, CLASSES) #해당 클래스 인덱스를 1로 두고 나머진 0으로 둬서 후에 해당 결과 값에대한 확률 계산 (ex :cls가 0 이면 [1,0,0,0,0,0,0,0])

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

            #validation : 학습에 대한 체크를 하기위한 과정 ,Test전 Overfitting을 예방하기위한 과정  (test에서는 train 과정과 다른 Data를 사용하기 때문)
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
                        
                print('final test set accuracy :', _accuracy / _accuracy_sample_cnt)
