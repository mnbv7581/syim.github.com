import tensorflow as tf
import numpy as np

import os
import cv2

from Learnit.SSD.Define import *
from Learnit.SSD.SSD import *
from Learnit.SSD.utils import *

#smooth L1
def smooth_L1(x):
    return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))

def ssd_loss(feature_classes, feature_locations, groundtruth_classes, groundtruth_locations, groundtruth_positives, groundtruth_count):
    #cross-entropy(pred class, ground truth class)
    softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_classes, labels=groundtruth_classes)

    #location 
    loss_location = tf.div(tf.reduce_sum(tf.multiply(
        #smooth L1
        tf.reduce_sum(smooth_L1(tf.subtract(groundtruth_locations, feature_locations)),
                      reduction_indices=2), groundtruth_positives), reduction_indices=1),
        #div
        tf.reduce_sum(groundtruth_positives, reduction_indices=1))

    #class
    loss_class = tf.div(
        tf.reduce_sum(tf.multiply(softmax_cross_entropy, groundtruth_count), reduction_indices=1),
        tf.reduce_sum(groundtruth_count, reduction_indices=1))

    #class loss, location loss * 5 = loss
    loss_all = tf.reduce_sum(tf.add(loss_class*1, loss_location*5))
    return loss_all, loss_class, loss_location

if __name__ == '__main__':
    input_image = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])

    feature_classes, feature_locations, all_default_boxes = SSD(input_image)

    groundtruth_classes = tf.placeholder(shape=[None, len(all_default_boxes)], dtype=tf.int32, name='groundtruth_classes')
    groundtruth_locations = tf.placeholder(shape=[None, len(all_default_boxes), 4], dtype=tf.float32, name='groundtruth_locations')
    groundtruth_positives = tf.placeholder(shape=[None, len(all_default_boxes)], dtype=tf.float32, name='groundtruth_positives')
    groundtruth_negatives = tf.placeholder(shape=[None, len(all_default_boxes)], dtype=tf.float32, name='groundtruth_negatives')
    groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)

    print(feature_classes)
    print(feature_locations)
    print(np.asarray(all_default_boxes).shape)

    total_loss, loss_class, loss_location = ssd_loss(feature_classes, feature_locations, groundtruth_classes, groundtruth_locations, groundtruth_positives, groundtruth_count)

    learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/ssd_{}.ckpt".format(105900))

        if True:
            image_names = os.listdir('../../DB/VOC/image/')

            for image_name in image_names:
                img = cv2.imread('../../DB/VOC/image/' + image_name)
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

                #input_img = (img - np.mean(img)) / np.std(img)
                input_img = img

                pred_feature_classes, pred_feature_locations = sess.run([feature_classes, feature_locations], feed_dict={input_image:[input_img]})

                dic_objects = pred_Decode(pred_feature_classes[0], pred_feature_locations[0], all_default_boxes)
                print(dic_objects)

                dic_objects['bbox'] = non_max_suppression_fast(dic_objects['bbox'], 0.75)

                for bbox in dic_objects['bbox']:
                    xmin, ymin, xmax, ymax = bbox
                    cv2.rectangle(input_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
                cv2.imshow('show', input_img)
                cv2.waitKey(0)

        epoch_learning_rate = 1e-4
        for epoch in range(MAX_EPOCHS):
            if epoch == (MAX_EPOCHS * 0.5) or epoch == (MAX_EPOCHS * 0.75):
                epoch_learning_rate /= 10

            img_data, xml_data = get_traindata_voc(BATCH_SIZE)
            #print(xml_data[0])

            gt_classes, gt_locations, gt_positives, gt_negatives = Encode(xml_data, all_default_boxes)

            '''
            dic_objects = Decode(gt_classes[0], gt_locations[0], all_default_boxes)
            print(dic_objects)

            for bbox in dic_objects['bbox']:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(img_data[0], (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            cv2.imshow('show', img_data[0])
            cv2.waitKey(0)
            '''

            _total_loss, _loss_classes, _loss_locations, _ = sess.run([total_loss, loss_class, loss_location, train_op],feed_dict={input_image:img_data, groundtruth_classes:gt_classes, groundtruth_locations:gt_locations, groundtruth_positives:gt_positives, groundtruth_negatives:gt_negatives, learning_rate:epoch_learning_rate})
            
            if epoch % 100 == 0:
                print('total loss : {}, loss class : {}, loss location : {}'.format(_total_loss, sum(_loss_classes)/BATCH_SIZE, sum(_loss_locations)/BATCH_SIZE))
                saver.save(sess, './model/ssd_{}.ckpt'.format(epoch))

