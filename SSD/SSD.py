import numpy as np
import tensorflow as tf

from Define import *

def BatchNormalization(x, scope):
    return tf.contrib.layers.batch_norm(inputs = x, scope=scope)

def conv_bn_relu(x, num_filter, layer_name, kernel_size = [3, 3], strides = 1, pad_option = 'SAME'):
    
    x = tf.layers.conv2d(inputs = x, filters = num_filter, kernel_size = kernel_size, strides = strides, padding = pad_option, name = layer_name + '_conv_1')
    x = BatchNormalization(x, layer_name + '_bn_1')
    x = tf.nn.relu(x, name = layer_name + '_relu_1')

    return x

def VGG16(input):

    filters = [32, 64, 128, 256]

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
    for i in range(3):
        x = conv_bn_relu(x, filters[2], 'vgg_3_' + str(i))
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_3_pool_1')
    print(x)

    #block 4
    for i in range(3):
        x = conv_bn_relu(x, filters[3], 'vgg_4_' + str(i))
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_4_pool_1')
    print(x)

    #block 5
    for i in range(3):
        x = conv_bn_relu(x, filters[3], 'vgg_5_' + str(i))
    #x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_5_pool_1')
    print(x)

    return x

def SSD(input):

    SSD_features = [512, 256, 128, 64]
    feature_maps = []

    x = VGG16(input)

    ssd_x = tf.layers.conv2d(inputs = x, filters = len( DEFAULT_BOX_ASPECT_RATIO[0]) * (CLASSES + 4), kernel_size = [3, 3], strides = 1, padding = 'SAME', name = 'SSD_conv_1')
    feature_maps.append(ssd_x)
    print(ssd_x)

    x = conv_bn_relu(x, SSD_features[0] / 2, 'SSD_1_', [3, 3])
    x = conv_bn_relu(x, SSD_features[0], 'SSD_2_', [1, 1])

    ssd_x = tf.layers.conv2d(inputs = x, filters = len( DEFAULT_BOX_ASPECT_RATIO[0]) * (CLASSES + 4), kernel_size = [3, 3], strides = 1, padding = 'SAME', name = 'SSD_conv_2')
    feature_maps.append(ssd_x)
    print(ssd_x)

    x = conv_bn_relu(x, SSD_features[1], 'SSD_3_', [1, 1])
    x = conv_bn_relu(x, SSD_features[1] / 2, 'SSD_4_', [3, 3], 2)

    ssd_x = tf.layers.conv2d(inputs = x, filters = len( DEFAULT_BOX_ASPECT_RATIO[0]) * (CLASSES + 4), kernel_size = [3, 3], strides = 1, padding = 'SAME', name = 'SSD_conv_3')
    feature_maps.append(ssd_x)
    print(ssd_x)

    x = conv_bn_relu(x, SSD_features[2], 'SSD_5_', [1, 1])
    x = conv_bn_relu(x, SSD_features[2] / 2, 'SSD_6_', [3, 3], 2)

    ssd_x = tf.layers.conv2d(inputs = x, filters = len( DEFAULT_BOX_ASPECT_RATIO[0]) * (CLASSES + 4), kernel_size = [3, 3], strides = 1, padding = 'SAME', name = 'SSD_conv_4')
    feature_maps.append(ssd_x)
    print(ssd_x)

    x = conv_bn_relu(x, SSD_features[3], 'SSD_7_', [1, 1])
    x = conv_bn_relu(x, SSD_features[3] / 2, 'SSD_8_', [3, 3], 2)

    ssd_x = tf.layers.conv2d(inputs = x, filters = len( DEFAULT_BOX_ASPECT_RATIO[0]) * (CLASSES + 4), kernel_size = [3, 3], strides = 1, padding = 'SAME', name = 'SSD_conv_5')
    feature_maps.append(ssd_x)
    print(ssd_x)

    x = conv_bn_relu(x, SSD_features[3], 'SSD_9_', [1, 1])
    x = conv_bn_relu(x, SSD_features[3] / 2, 'SSD_10_', [3, 3], 1, 'valid')

    ssd_x = tf.layers.conv2d(inputs = x, filters = len( DEFAULT_BOX_ASPECT_RATIO[0]) * (CLASSES + 4), kernel_size = [3, 3], strides = 1, padding = 'SAME', name = 'SSD_conv_6')
    feature_maps.append(ssd_x)
    print(ssd_x)

    tmp_all_features = []
    all_default_boxes = []

    feature_maps_shape = [m.get_shape().as_list() for m in feature_maps]

    for i in range(len(feature_maps)):
        f_w = feature_maps_shape[i][1]
        f_h = feature_maps_shape[i][2]

        tmp_all_features.append(tf.reshape(feature_maps[i], [-1, (f_w * f_h * len( DEFAULT_BOX_ASPECT_RATIO[0])), CLASSES + 4]))

        scale = DEFAULT_BOX_SCALE[i]
        ratios = DEFAULT_BOX_ASPECT_RATIO[i]

        for x in range(int(f_w)):
            for y in range(int(f_h)):
                for ratio in ratios:

                    cx = float((x + 0.5) / f_w)
                    cy = float((y + 0.5) / f_h)
                    w = float(scale * (ratio ** 0.5) / 1.2)
                    h = float(scale * (ratio ** 0.5) / 1.2)

                    #xmin = cx - w / 2
                    #ymin = cy - h / 2
                    #xmax = cx + w / 2
                    #ymax = cy - h / 2
                    all_default_boxes.append([cx, cy, w, h])
                
                #cx = float((x + 0.5) / w)
                #cy = float((y + 0.5) / h)
                #w = float(scale * 1.5)
                #h = float(scale * 1.4)
                #all_default_boxes.append([cx, cy, w, h])

    tmp_all_features = tf.concat(tmp_all_features, axis=1)
    print('tmp_all_features :', tmp_all_features)

    feature_classes = tmp_all_features[:, :, :CLASSES]
    feature_locations = tmp_all_features[:, :, CLASSES:]

    return feature_classes, feature_locations, all_default_boxes

if __name__ == '__main__':
    print(len( DEFAULT_BOX_ASPECT_RATIO[0]))
    input = tf.placeholder(tf.float32, [None, 300, 300, 3])
    SSD(input)