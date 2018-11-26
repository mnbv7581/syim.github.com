import os

import numpy as np
from Define import *

import skimage.io
import skimage.transform

import imutils
import xml.etree.ElementTree as etxml

import os
import gc
import xml.etree.ElementTree as etxml
import math
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables
import time
from imutils.object_detection import non_max_suppression
import imutils
import cv2
import matplotlib.pyplot as plt

img_names = os.listdir(IMG_PATH)
print('image load : {}'.format(len(img_names)))

def get_actual_data_from_xml(xml_path):

    #print(xml_path)

    actual_item = []
    #try:
    if True:
        annotation_node = etxml.parse(xml_path).getroot()
        img_width = float(annotation_node.find('size').find('width').text.strip())
        img_height = float(annotation_node.find('size').find('height').text.strip())
        object_node_list = annotation_node.findall('object')
        for obj_node in object_node_list:
            lable = LABELS[obj_node.find('name').text]
            bndbox = obj_node.find('bndbox')
            x_min = float(bndbox.find('xmin').text.strip())
            y_min = float(bndbox.find('ymin').text.strip())
            x_max = float(bndbox.find('xmax').text.strip())
            y_max = float(bndbox.find('ymax').text.strip())

            #normalize (0 ~ 1.0)
            #[cx, cy, w, h, label]
            actual_item.append([((x_min + x_max) / 2 / img_width), ((y_min + y_max) / 2 / img_height),
                                ((x_max - x_min) / img_width), ((y_max - y_min) / img_height), lable])
        return actual_item
    #except:
    #    return None

jitter = 0.2
def get_traindata_voc(batch_size):

    img_data = []
    xml_data = []

    file_list = random.sample(img_names, batch_size)

    for f_name in file_list:
        img_path = IMG_PATH + f_name
        xml_path = XML_PATH + f_name.replace('.jpg', '.xml')

        if os.path.splitext(img_path)[1].lower() == '.jpg':

            _img = cv2.imread(img_path)
            _img = cv2.resize(_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
            #img /= 255.

            img = (_img - np.mean(_img)) / np.std(_img)

            xml = get_actual_data_from_xml(xml_path)
            #print(f_name, xml)
            #input()

            img_data.append(img)
            xml_data.append(xml)

    return img_data, xml_data

def jaccard(rect1, rect2):
    x_overlap = max(0, ( min(rect1[0] + (rect1[2] / 2), rect2[0] + (rect2[2] / 2)) - max(rect1[0] - (rect1[2] / 2), rect2[0] - (rect2[2] / 2)) ))
    y_overlap = max(0, ( min(rect1[1] + (rect1[3] / 2), rect2[1] + (rect2[3] / 2)) - max(rect1[1] - (rect1[3] / 2), rect2[1] - (rect2[3] / 2)) ))

    intersection = x_overlap * y_overlap

    rect1_width_sub = 0
    rect1_height_sub = 0
    rect2_width_sub = 0
    rect2_height_sub = 0

    if (rect1[0] - rect1[2] / 2) < 0: rect1_width_sub += 0 - (rect1[0] - rect1[2] / 2)
    if (rect1[0] + rect1[2] / 2) > 1: rect1_width_sub += (rect1[0] + rect1[2] / 2) - 1
    if (rect1[1] - rect1[3] / 2) < 0: rect1_height_sub += 0 - (rect1[1] - rect1[3] / 2)
    if (rect1[1] + rect1[3] / 2) > 1: rect1_height_sub += (rect1[1] + rect1[3] / 2) - 1
    if (rect2[0] - rect2[2] / 2) < 0: rect2_width_sub += 0 - (rect2[0] - rect2[2] / 2)
    if (rect2[0] + rect2[2] / 2) > 1: rect2_width_sub += (rect2[0] + rect2[2] / 2) - 1
    if (rect2[1] - rect2[3] / 2) < 0: rect2_height_sub += 0 - (rect2[1] - rect2[3] / 2)
    if (rect2[1] + rect2[3] / 2) > 1: rect2_height_sub += (rect2[1] + rect2[3] / 2) - 1

    area_box_a = (rect1[2] - rect1_width_sub) * (rect1[3] - rect1_height_sub)
    area_box_b = (rect2[2] - rect2_width_sub) * (rect2[3] - rect2_height_sub)
    union = area_box_a + area_box_b - intersection

    if intersection > 0 and union > 0:
        return intersection / union,[(rect1[0]-(rect2[0]))/rect2[2],(rect1[1]-(rect2[1]))/rect2[3],math.log(rect1[2]/rect2[2]),math.log(rect1[3]/rect2[3])]
    else:
        return 0,[0.00001,0.00001,0.00001,0.00001]

def Encode(input_data, all_default_boxes):

    assert BATCH_SIZE == len(input_data)
    assert all_default_boxes != 0

    gt_classes = np.zeros((BATCH_SIZE, len(all_default_boxes)), dtype = np.float32)
    gt_locations = np.zeros((BATCH_SIZE, len(all_default_boxes), 4), dtype = np.float32)

    gt_positives_jacc = np.zeros((BATCH_SIZE, len(all_default_boxes)), dtype = np.float32)
    gt_positives = np.zeros((BATCH_SIZE, len(all_default_boxes)), dtype = np.float32)
    gt_negatives = np.zeros((BATCH_SIZE, len(all_default_boxes)), dtype = np.float32)

    background_jacc = max(0, (JACCARD_VALUE - 0.3))

    for batch_index in range(BATCH_SIZE):

        # create positive
        # data = [cx, cy, w, h, class]
        for data in input_data[batch_index]:

            #get class
            gt_class = data[-1]
            assert gt_class >= 0 and gt_class < CLASSES

            #get bounding box
            gt_box = data[:-1]
            for box_index in range(len(all_default_boxes)):
                jacc, gt_box_loc = jaccard(gt_box, all_default_boxes[box_index])

                # iou >= threshold
                if jacc >= JACCARD_VALUE:
                    # set ground truth
                    gt_classes[batch_index][box_index] = gt_class
                    gt_locations[batch_index][box_index] = gt_box_loc

                    gt_positives_jacc[batch_index][box_index] = jacc
                    gt_positives[batch_index][box_index] = 1
                    gt_negatives[batch_index][box_index] = 0

        # object check
        if np.sum(gt_positives[batch_index]) == 0:
            random_pos_index = np.random.randint(low = 0, high = len(all_default_boxes), size = 1)[0]
            gt_classes[batch_index][random_pos_index] = BACKGROUND_CLASS
            gt_locations[batch_index][random_pos_index] = [0.00001, 0.00001, 0.00001, 0.00001]
            
            gt_positives_jacc[batch_index][random_pos_index] = JACCARD_VALUE
            gt_positives[batch_index][random_pos_index] = 1
            gt_negatives[batch_index][random_pos_index] = 0

        # positive + negative check
        gt_positives_count = np.sum(gt_positives[batch_index])
        gt_negatives_count = int(gt_positives_count * 3)
        if (gt_negatives_count + gt_positives_count > len(all_default_boxes)):
            gt_negatives_count = len(all_default_boxes) - gt_positives_count

        # create negative
        gt_negatives_index = np.random.randint(low = 0, high = len(all_default_boxes), size = gt_negatives_count)
        for negative_index in gt_negatives_index:
            if gt_positives_jacc[batch_index][negative_index] < background_jacc and gt_positives[batch_index][negative_index] == 0:
                gt_classes[batch_index][negative_index] = BACKGROUND_CLASS
                gt_positives[batch_index][negative_index] = 0
                gt_negatives[batch_index][negative_index] = 1

    return gt_classes, gt_locations, gt_positives, gt_negatives

'''
#actual data -> label
def generate_groundtruth_data(input_actual_data, all_default_boxs):

    #batch size
    input_actual_data_len = len(input_actual_data)
    all_default_boxs_len = len(all_default_boxs)

    #batch size, anchor size, class
    gt_class = np.zeros((input_actual_data_len, all_default_boxs_len))

    #batch size, anchor, bounding box
    gt_location = np.zeros((input_actual_data_len, all_default_boxs_len, 4))

    #positive
    #batch size, anchor, jaccard overlap
    gt_positives_jacc = np.zeros((input_actual_data_len, all_default_boxs_len))
    #batch size, anchor, confidence ?
    gt_positives = np.zeros((input_actual_data_len, all_default_boxs_len))

    #negative
    #batch size, anchor
    gt_negatives = np.zeros((input_actual_data_len, all_default_boxs_len))

    #기준 jaccard threshold
    background_jacc = max(0, (JACCARD_VALUE - 0.2))

    #image index <= batch size
    for img_index in range(input_actual_data_len):

        #actual = [cx, cy, w, h, label]
        for pre_actual in input_actual_data[img_index]:

            #class
            gt_class_val = pre_actual[-1:][0]
            if gt_class_val>20 or gt_class_val<0:
                gt_class_val=0

            #bounding box
            gt_box_val = pre_actual[:-1]
            #box index <= anchor size
            for boxe_index in range(all_default_boxs_len):
                #jacc, gt_box_location = jaccard overlap (gt, default box)
                jacc,gt_box_val_loc = jaccard(gt_box_val, all_default_boxs[boxe_index])

                #jacc >= jaccard threshold
                if jacc > JACCARD_VALUE or jacc == JACCARD_VALUE:
                    gt_class[img_index][boxe_index] = gt_class_val
                    gt_location[img_index][boxe_index] = gt_box_val_loc
                    gt_positives_jacc[img_index][boxe_index] = jacc
                    gt_positives[img_index][boxe_index] = 1
                    gt_negatives[img_index][boxe_index] = 0 #?

        #object check
        if np.sum(gt_positives[img_index]) == 0:
            #no object
            random_pos_index = np.random.randint(low=0, high=all_default_boxs_len, size=1)[0]
            gt_class[img_index][random_pos_index] = background_classes_val
            gt_location[img_index][random_pos_index] = [0.00001, 0.00001, 0.00001, 0.00001]
            gt_positives_jacc[img_index][random_pos_index] = jaccard_value
            gt_positives[img_index][random_pos_index] = 1
            gt_negatives[img_index][random_pos_index] = 0 #?

        #positive * 3 => negative samples
        gt_neg_end_count = int(np.sum(gt_positives[img_index]) * 3)
        #all boxes < (neg + positive) check
        if (gt_neg_end_count + np.sum(gt_positives[img_index])) > all_default_boxs_len:
            #neg = all boxes - positive
            gt_neg_end_count = all_default_boxs_len - np.sum(gt_positives[img_index])

        #negative random index
        gt_neg_index = np.random.randint(low=0, high=all_default_boxs_len, size=gt_neg_end_count)
        for r_index in gt_neg_index:
            #jaccard overlap < background jaccard && not positive
            if gt_positives_jacc[img_index][r_index] < background_jacc and gt_positives[img_index][r_index] != 1:
                gt_class[img_index][r_index] = 0
                gt_positives[img_index][r_index] = 0
                gt_negatives[img_index][r_index] = 1
    #check value (nan check)
    gt_class = check_numerics(gt_class, 'gt_class')
    gt_location = check_numerics(gt_location, 'gt_class')
    gt_positives = check_numerics(gt_positives, 'gt_positives')
    gt_negatives = check_numerics(gt_negatives, 'gt_negatives')
    return gt_class, gt_location, gt_positives, gt_negatives
'''

def Decode(pred_classes, pred_locations, all_default_boxes):

    assert len(all_default_boxes) == len(pred_locations)

    dic_objects = {}
    dic_objects['bbox'] = []
    dic_objects['class'] = []

    for i in range(len(pred_locations)):
    
        #background
        if pred_classes[i] == BACKGROUND_CLASS:
            continue

        # pred_location = [cx, cy, w, h]
        # all_default_box = [cx, cy, w, h]
        cx = pred_locations[i][0] * all_default_boxes[i][2] + all_default_boxes[i][0]
        cy = pred_locations[i][1] * all_default_boxes[i][3] + all_default_boxes[i][1]
        w = all_default_boxes[i][2] * np.exp(pred_locations[i][2])
        h = all_default_boxes[i][3] * np.exp(pred_locations[i][3])

        xmin = int((cx - w / 2) * IMAGE_WIDTH)
        ymin = int((cy - h / 2) * IMAGE_HEIGHT)
        xmax = int((cx + w / 2) * IMAGE_WIDTH)
        ymax = int((cy + h / 2) * IMAGE_HEIGHT)

        # + min, max exception

        dic_objects['class'].append(pred_classes[i])
        dic_objects['bbox'].append([xmin, ymin, xmax, ymax])

    return dic_objects

def softmax(arr):
    arr -= np.max(arr)
    soft = np.exp(arr) / np.sum(np.exp(arr))

    return soft

def non_max_suppression_fast(boxes, overlapThresh):

	boxes = np.asarray(boxes)

	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def pred_Decode(pred_classes, pred_locations, all_default_boxes):

    assert len(all_default_boxes) == len(pred_locations)

    dic_objects = {}
    dic_objects['bbox'] = []
    dic_objects['class'] = []

    for i in range(len(pred_locations)):
    
        #background
        if np.argmax(pred_classes[i]) == BACKGROUND_CLASS:
            continue

        cls_idx = np.argmax(pred_classes[i])
        softmax_cls = softmax(pred_classes[i])

        if softmax_cls[cls_idx] < 0.5:
            continue

        # pred_location = [cx, cy, w, h]
        # all_default_box = [cx, cy, w, h]
        cx = pred_locations[i][0] * all_default_boxes[i][2] + all_default_boxes[i][0]
        cy = pred_locations[i][1] * all_default_boxes[i][3] + all_default_boxes[i][1]
        w = all_default_boxes[i][2] * np.exp(pred_locations[i][2])
        h = all_default_boxes[i][3] * np.exp(pred_locations[i][3])

        xmin = int((cx - w / 2) * IMAGE_WIDTH)
        ymin = int((cy - h / 2) * IMAGE_HEIGHT)
        xmax = int((cx + w / 2) * IMAGE_WIDTH)
        ymax = int((cy + h / 2) * IMAGE_HEIGHT)

        # + min, max exception

        if xmin < 0 or xmin > IMAGE_WIDTH or ymin < 0 or ymin > IMAGE_HEIGHT or xmax < 0 or xmax > IMAGE_WIDTH or ymax < 0 or ymax > IMAGE_HEIGHT:
            continue

        dic_objects['class'].append(np.argmax(pred_classes[i]))
        dic_objects['bbox'].append([xmin, ymin, xmax, ymax])

    return dic_objects