
import numpy as np
import tensorflow as tf

from Learnit.VGG.Define import *

def BatchNormalization(x, scope): #Batch로 들어온 입력에 대해서 노멀라이제이션
    return tf.contrib.layers.batch_norm(inputs = x, scope=scope)

def conv_bn_relu(x, num_filter, layer_name): # 입력, 필터수 지정, 라벨 이름
    
    x = tf.layers.conv2d(inputs = x, filters = num_filter, kernel_size = [3, 3], strides = 1, padding = 'SAME', name = layer_name + '_conv_1')
    x = BatchNormalization(x, layer_name + '_bn_1')#데이터의 쏠림현상 방지, 분산 또는 편차를 이용해서 데이터가 한쪽으로 치우치는 것을 방치(이미지 상에선 같은 특징이지만 실제 Value 값이 차이가 다른 파라미터를 표준화 시켜준다) 
    x = tf.nn.relu(x, name = layer_name + '_relu_1')#데이터(W)를 살릴지 죽일지는 정하는 과정

    return x

def VGG16(input, training_flag): #
    x = input
    print(x)

    #block 1
    for i in range(2):
        x = conv_bn_relu(x, filters[0], 'vgg_1_' + str(i))
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_1_pool_1')
    print(x)
    
    #block 2
    for i in range(2):
        x = conv_bn_relu(x, filters[1], 'vgg_2_' + str(i))
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_2_pool_1')
    print(x)

    #block 3
    for i in range(5):
        x = conv_bn_relu(x, filters[2], 'vgg_3_' + str(i))
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_3_pool_1')
    print(x)

    #block 4
    for i in range(5):
        x = conv_bn_relu(x, filters[3], 'vgg_4_' + str(i))
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_4_pool_1')
    print(x)

    #block 5
    for i in range(5):
        x = conv_bn_relu(x, filters[3], 'vgg_5_' + str(i))
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_5_pool_1')
    print(x)

    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(inputs = x, units = fc_parameters[0], name = 'fc_1')
    x = tf.layers.dropout(inputs = x, rate = DROP_OUT, training = training_flag)

    x = tf.layers.dense(inputs = x, units = fc_parameters[1], name = 'fc_2')
    x = tf.layers.dropout(inputs = x, rate = DROP_OUT, training = training_flag)

    x = tf.layers.dense(inputs = x, units = CLASSES, name = 'fc_3')
        
    return x
