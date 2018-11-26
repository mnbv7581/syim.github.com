
import os
import cv2
from time import time

import random
import numpy as np
import tensorflow as tf

from YOLOv2 import *
from Define import *
from utils import *

from utils import _decode_yolov2_output

from tensorflow.python.platform import app
from tensorflow.python.summary import summary
from tensorflow.python.framework import graph_util

def yolo_loss(preds, labels):
	mask = slice_tensor(labels, 5) #confidence
	labels = slice_tensor(labels, 0, 4) #x, y, w, h, class
	
	mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)), tf.bool)
		 
	with tf.name_scope('mask'):
		masked_label = tf.boolean_mask(labels, mask)
		masked_pred = tf.boolean_mask(preds, mask)
		neg_masked_pred = tf.boolean_mask(preds, tf.logical_not(mask))

	with tf.name_scope('pred'):
		masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
		masked_pred_wh = tf.exp(slice_tensor(masked_pred, 2, 3))
		masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 4))
		masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 4))
		masked_pred_c = tf.nn.softmax(slice_tensor(masked_pred, 5, -1))
		
	with tf.name_scope('lab'):
		masked_label_xy = slice_tensor(masked_label, 0, 1)
		masked_label_wh = slice_tensor(masked_label, 2, 3)
		masked_label_c = slice_tensor(masked_label, 4)
		masked_label_c_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_c, tf.int32), depth=N_CLASSES), shape=(-1, N_CLASSES))

	with tf.name_scope('total_loss'):
		with tf.name_scope('loss_xy'):
			loss_xy = tf.reduce_sum(tf.square(masked_pred_xy-masked_label_xy))
		with tf.name_scope('loss_wh'):	
			loss_wh = tf.reduce_sum(tf.square(masked_pred_wh-masked_label_wh))
		with tf.name_scope('loss_obj'):
			loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1))
		with tf.name_scope('loss_no_obj'):
			loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o))
		with tf.name_scope('loss_class'):	
			loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))
		
		loss = tf.reduce_mean(COORD*(loss_xy + loss_wh) + loss_obj + NOOBJ * loss_no_obj + loss_c)
	return loss

if __name__ == '__main__':

    xml_names = os.listdir(DB_XML_DIRS[0])
    xml_names = np.asarray(xml_names)
    length = len(xml_names)

    train_length = int(length * 0.9)

    train_xml_names = xml_names[:train_length]
    test_xml_names  = xml_names[train_length:]
    print(len(train_xml_names))
    print(len(test_xml_names))
    model_path = './test_model/yolov2_{}.ckpt'

    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name = 'input')
    label_var = tf.placeholder(tf.float32, [None, GRID_H, GRID_W, N_ANCHORS, BOX_SIZE + 1], name = 'label')

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    add = tf.add(X, Y)
    mul = tf.multiply(X, Y)

    # step 1: node 선택
    add_hist = tf.summary.scalar('add_scalar', add)
    mul_hist = tf.summary.scalar('mul_scalar', mul)

    # step 2: summary 통합. 두 개의 코드 모두 동작.
    merged = tf.summary.merge_all()
    # merged = tf.summary.merge([add_hist, mul_hist])

    
    isTraining = False #Training Flag
    with tf.variable_scope('network'):
        yolov2 = YOLOv2(input_var, isTraining)

    with tf.name_scope('loss'):
        loss = yolo_loss(yolov2, label_var)

    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        restore_index = 1000001

        sess.run(tf.global_variables_initializer())

        if restore_index != 0:
            saver.restore(sess, model_path.format(restore_index))

        #frozen graph save
        #if not isTraining:
        if isTraining:
            gd = sess.graph.as_graph_def()
            converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ['network/outputs'])
            tf.train.write_graph(converted_graph_def, './', 'yolov2_graph.pb', as_text=False)
            input('freeze graph save complete')

        #tensorboard visualize
        if isTraining:
            writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR, graph = sess.graph)
            #writer.close()  

            print("Model Imported. Visualize by running: tensorboard --logdir={}".format(TENSORBOARD_LOGDIR))

        # graph save
        if isTraining:
            tf.train.write_graph(sess.graph, './model/', 'yolov2_graph.pbtxt', as_text = True)
            tf.train.write_graph(sess.graph, './model/', 'yolov2_graph.pb', as_text = False)
            input('txt graph, graph save complete')

        if not isTraining:
            #image_names = os.listdir('./test_db/image/')
            image_names = os.listdir(DB_IMG_DIRS2[0])

            for image_name in image_names:
                #print('./test_db/image/' + image_name)
                #img = cv2.imread('./test_db/image/' + image_name)
                print(DB_IMG_DIRS2[0] + image_name)
                img = cv2.imread(DB_IMG_DIRS2[0] + image_name)
                original_shape = img.shape

                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                change_shape = img.shape
                
                output = sess.run(yolov2, feed_dict={input_var:[img]})[0]
                objects = _decode_yolov2_output(output, 0.25)
                
                #img_path, gt_bboxes, gt_classes = xml_read("./test_db/xml/" + image_name[:-4] + '.xml', 'xyxy')
                img_path, gt_bboxes, gt_classes = xml_read(DB_XML_DIRS2[0] + image_name[:-4] + '.xml', 'xyxy')
                
                gt_bboxes = convert_bboxes_scale(gt_bboxes, original_shape, change_shape)

                pred_boxes = []
                pred_classes = []

                for object in objects:
                    bbox, cls = object
                    
                    pred_boxes.append(bbox)
                    pred_classes.append(cls)
                
                #map = mAP(gt_bboxes, gt_classes, pred_boxes, pred_classes)

                print('# Prediction')
                for i in range(len(pred_boxes)):
                    xmin, ymin, xmax, ymax = pred_boxes[i]
                    cls = pred_classes[i]

                    print(cls, pred_boxes[i])
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                print()

#===============================================================================
#                 print('# GT')
#                 for i in range(len(gt_bboxes)):
#                     xmin, ymin, xmax, ymax = gt_bboxes[i]
#                     cls_idx = gt_classes[i]
# 
#                     print(CLASS_NAME[cls_idx], gt_bboxes[i])
#                     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
#                 print()
#===============================================================================

                cv2.imshow('show', img)
                cv2.waitKey(0)

                #mAPs.append(map)

            #input('mAP : {}'.format(np.mean(mAPs)))
            import sys
            sys.exit(0)

        st = time()
        iter_losses = []

        iter_learning_rate = 1e-4 
        for iter in range(1 + restore_index, N_ITERS + 1):
            if iter == int(N_ITERS * 0.75) or iter == int(N_ITERS * 0.5):
                iter_learning_rate /= 10

            random.shuffle(train_xml_names)
            np_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), np.float32)
            np_label_data = np.zeros((BATCH_SIZE, GRID_H, GRID_W, N_ANCHORS, BOX_SIZE + 1), np.float32)

            train_data = get_generate_yolov2_labels(DB_XML_DIRS[0], train_xml_names[:BATCH_SIZE])

            for i in range(BATCH_SIZE):
                image_data, label_data = train_data[i]
                np_image_data[i] = image_data
                np_label_data[i] = label_data

                #image_data = image_data.astype(np.uint8)
                #print(image_data.shape)

                #cv2.imshow('original', image_data)
                #objects = decode_yolov2_output(label_data, 0.5)
                #print(objects)

                #for object in objects:
                #    bbox, cls = object
                #    xmin, ymin, xmax, ymax = bbox
                #    cv2.rectangle(image_data, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                #cv2.imshow('show', image_data)
                #cv2.waitKey(0)

            _, iter_loss = sess.run([train_op, loss], feed_dict = {input_var : np_image_data,
                                                                   label_var : np_label_data,
                                                                   learning_rate : iter_learning_rate})
            iter_losses.append(iter_loss)

            if iter % LOG_ITER == 0:
                log_time = time() - st

                print('iter : %d | time : %.2f | loss : %.4f'%(iter, log_time, np.mean(iter_losses)))

                st = time()
                iter_losses = []

            if iter % SAVE_ITER == 0:
                saver.save(sess, model_path.format(iter))
                #test mAP
writer.close()  