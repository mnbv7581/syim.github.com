import numpy as np
import tensorflow as tf

from Define import *

def BatchNormalization(x, training, scope):
    return tf.layers.batch_normalization(x, training=training, name = scope)

def conv_bn_relu(x, kernel_size, num_filters, training_flag, layer_name):
    init = tf.contrib.layers.xavier_initializer()

    x = tf.layers.conv2d(inputs = x, filters = num_filters, kernel_size = kernel_size, kernel_initializer = init, strides = 1, padding = 'SAME', name = layer_name)
    x = BatchNormalization(x, training_flag, layer_name + '_bn')
    x = tf.nn.leaky_relu(x)

    return x

def residual_block_first(x, out_channel, layer_name, isTraining):
    conv_index = 1
    
    #x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = layer_name + '_pool_1')
    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 2, padding = 'same', name = layer_name + '_down_conv_' + str(conv_index))
    conv_index += 1

    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    last_x = x

    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    x = BatchNormalization(x, isTraining, layer_name + '_bn_1')
    x = tf.nn.leaky_relu(x, name = layer_name + '_relu_1')
    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    x = BatchNormalization(x, isTraining, layer_name + '_bn_2')
    
    x = x + last_x
    x = tf.nn.leaky_relu(x, name = layer_name + '_relu_2')

    return x

def residual_block(x, layer_name, isTraining):
    num_channel = x.get_shape().as_list()[-1]

    last_x = x
    x = tf.layers.conv2d(inputs = x, filters = num_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_1')
    x = BatchNormalization(x, isTraining, layer_name + '_bn_1')
    x = tf.nn.leaky_relu(x, name = layer_name + '_relu_1')
    x = tf.layers.conv2d(inputs = x, filters = num_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_2')
    x = BatchNormalization(x, isTraining, layer_name + '_bn_2')

    x = x + last_x
    x = tf.nn.leaky_relu(x, name = layer_name + '_relu_2')

    return x

def passthrough_layer(a, b, kernel, depth, size, training_flag, name):
	
	b = conv_bn_relu(b, kernel, depth, training_flag,name)
	b = tf.space_to_depth(b, size)
	y = tf.concat([a, b], axis=3)
	
	return y

#def YOLOv2(x, training_flag):

#	x = conv_bn_relu(x, (3, 3), 32, training_flag, 'conv1')
#	x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool1')
#	x = conv_bn_relu(x, (3, 3), 64, training_flag, 'conv2')
#	x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool2')
	
#	x = conv_bn_relu(x, (3, 3), 128, training_flag, 'conv3')
#	x = conv_bn_relu(x, (1, 1), 64, training_flag, 'conv4')
#	x = conv_bn_relu(x, (3, 3), 128, training_flag, 'conv5')
#	x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool5')

#	x = conv_bn_relu(x, (3, 3), 256, training_flag, 'conv6')
#	x = conv_bn_relu(x, (1, 1), 128, training_flag, 'conv7')
#	x = conv_bn_relu(x, (3, 3), 256, training_flag, 'conv8')
#	x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool8')

#	x = conv_bn_relu(x, (3, 3), 512, training_flag, 'conv9')
#	x = conv_bn_relu(x, (1, 1), 256, training_flag, 'conv10')
#	x = conv_bn_relu(x, (3, 3), 512, training_flag, 'conv11')
#	x = conv_bn_relu(x, (1, 1), 256, training_flag, 'conv12')
#	passthrough = conv_bn_relu(x, (3, 3), 512, training_flag, 'conv13')
#	x = tf.layers.max_pooling2d(inputs = passthrough, pool_size = [2, 2], strides = 2, name = 'maxpool13')
	
#	x = conv_bn_relu(x, (3, 3), 1024, training_flag, 'conv14')
#	x = conv_bn_relu(x, (1, 1), 512, training_flag, 'conv15')
#	x = conv_bn_relu(x, (3, 3), 1024, training_flag, 'conv16')
#	x = conv_bn_relu(x, (1, 1), 512, training_flag, 'conv17')
#	x = conv_bn_relu(x, (3, 3), 1024, training_flag, 'conv18')

#	x = conv_bn_relu(x, (3, 3), 1024, training_flag, 'conv19')
#	x = conv_bn_relu(x, (3, 3), 1024, training_flag, 'conv20')
#	x = passthrough_layer(x, passthrough, (3, 3), 64, 2, training_flag, 'conv21')					 
#	x = conv_bn_relu(x, (3, 3), 1024, training_flag, 'conv22')
#	x = conv_bn_relu(x, (1, 1), N_ANCHORS * (N_CLASSES + 5), training_flag, 'conv23')

#	x = tf.reshape(x, shape=(-1, GRID_H, GRID_W, N_ANCHORS, N_CLASSES + BOX_SIZE), name='outputs')					
#	return x

def YOLOv2(x, training_flag):

    #mean, variance = tf.nn.moments(x, axes = 0)
    #stddev = tf.sqrt(variance)
    #x = (x - mean)/stddev

    x = x / 127.5 - 1.0

    num_residual = 5

    x = tf.layers.conv2d(inputs = x, filters = filters[0], kernel_size = [7, 7], strides = 2, padding = 'SAME', name = 'resnet_1_conv_1')
    x = BatchNormalization(x, training_flag, 'resnet_1_bn_1')
    x = tf.nn.leaky_relu(x, name = 'resnet_1_relu_1')
    #x = tf.layers.conv2d(inputs = x, filters = filters[0], kernel_size = [3, 3], strides = 2, padding = 'same', name = 'resnet_1_down_conv_1')
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'resnet_1_pool_1')
    
    #print(x)
    
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

    x = residual_block_first(x, filters[3], layer_name='resnet_block_3', isTraining = training_flag)
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)

    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)
    
    x = tf.layers.conv2d(inputs = x, filters = N_ANCHORS * (N_CLASSES + 5), kernel_size = [1, 1], strides = 1, padding='same', name= 'conv22')
    x = BatchNormalization(x, training_flag, 'conv22_bn')

    x = tf.reshape(x, shape=(-1, GRID_H, GRID_W, N_ANCHORS, N_CLASSES + BOX_SIZE), name='outputs')					
    return x

if __name__ == '__main__':
    import numpy as np
    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

    yolov2 = YOLOv2(input_var, False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    outputs = sess.run(yolov2, feed_dict={input_var:np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))})
    print(outputs)
