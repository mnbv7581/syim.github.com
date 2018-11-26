import tensorflow as tf
import numpy as np

from Learnit.CAM.Define import *

def Global_Average_Pooling(x, stride=1) :
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]

    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) 

def BatchNormalization(x, training, scope):
    return tf.layers.batch_normalization(x, training=training, name = scope)

def residual_block_first(x, out_channel, layer_name, isTraining, size = 'valid'):
    conv_index = 1
    
    if size == 'valid':
        x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = layer_name + '_pool_1')
    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    last_x = x

    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    x = BatchNormalization(x, isTraining, layer_name + '_bn_1')
    x = tf.nn.relu(x, name = layer_name + '_relu_1')
    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    x = BatchNormalization(x, isTraining, layer_name + '_bn_2')
    
    x = x + last_x
    x = tf.nn.relu(x, name = layer_name + '_relu_2')

    return x

def residual_block(x, layer_name, isTraining):
    num_channel = x.get_shape().as_list()[-1]

    last_x = x
    x = tf.layers.conv2d(inputs = x, filters = num_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_1')
    x = BatchNormalization(x, isTraining, layer_name + '_bn_1')
    x = tf.nn.relu(x, name = layer_name + '_relu_1')
    x = tf.layers.conv2d(inputs = x, filters = num_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_2')
    x = BatchNormalization(x, isTraining, layer_name + '_bn_2')

    x = x + last_x
    x = tf.nn.relu(x, name = layer_name + '_relu_2')

    return x

def ResNet(input, training_flag, num_residual = 3):
    x = input / 255 - 0.5
    filters = [16, 32, 64, 128]

    x = BatchNormalization(x, training_flag, 'resnet_1_bn_1')
    
    print(x)
    
    resnet_index = 2
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)

    x = residual_block_first(x, filters[1], layer_name='resnet_block_1', isTraining = training_flag)
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)

    x = residual_block_first(x, filters[2], layer_name='resnet_block_2', isTraining = training_flag)
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)

    x = residual_block_first(x, filters[3], layer_name='resnet_block_3', isTraining = training_flag, size = 'same')
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x) 
    
    conv =  tf.layers.conv2d(inputs = x, filters = 128, kernel_size = [3, 3], strides = 1, padding = 'SAME', name = 'conv_features')
    x = conv
    
    x = Global_Average_Pooling(x)
    x = tf.contrib.layers.flatten(x)

    fc_initial = tf.Variable(tf.random_normal([128, CLASSES]), name = 'fc_Weight')
    
    x = tf.matmul(x, fc_initial, name = 'predict')
    print(x)

    return x, conv, fc_initial

def Visualize(cls_idx, conv, fc_w):
    heatmap_conv = tf.reshape(conv, [-1, 8*8, 128])
    heatmap_fc_w = tf.reshape(fc_w[:, cls_idx], [-1, 128, 1])
    heatmap_flat = tf.matmul(heatmap_conv, heatmap_fc_w, name = 'heatmap')

    heatmaps = tf.reshape(heatmap_flat, [-1, 8,8])
    return heatmapsds

if __name__ == '__main__':
    _input = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])

    resnet, conv, fc_initial = ResNet(_input, True)
    heatmaps = Visualize(0, conv, fc_initial)

    print(resnet)
    print(heatmaps)