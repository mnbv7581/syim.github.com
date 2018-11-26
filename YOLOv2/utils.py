
import os
import cv2

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from Define import *

def convert_bboxes_scale(bboxes, original_shape, change_shape):
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]

        xmin = int(xmin / original_shape[1] * change_shape[1])
        ymin = int(ymin / original_shape[0] * change_shape[0])
        xmax = int(xmax / original_shape[1] * change_shape[1])
        ymax = int(ymax / original_shape[0] * change_shape[0])

        bboxes[i] = [xmin, ymin, xmax, ymax]
    return bboxes

def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = (xB - xA) * (yB - yA)

    if (xB - xA) < 0.0:
        return 0.0

    if (yB - yA) < 0.0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if interArea < 0.0:
        return 0.0

    if (boxAArea + boxBArea - interArea) <= 0.0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return max(iou, 0.0)

def mAP(gt_boxes, gt_class_index, pred_boxes, pred_class_index, threshold_iou = 0.5):
    precision = 0.0
    recall = 0.0

    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return 1.0, 1.0
        else:
            return 0.0, 0.0

    if len(pred_boxes) != 0:
        gt_boxes_cnt = len(gt_boxes)
        pred_boxes_cnt = len(pred_boxes)

        recall_vector = []
        precision_vector = []

        for i in range(gt_boxes_cnt):
            recall_vector.append(False)
        for i in range(pred_boxes_cnt):
            precision_vector.append(False)

        for gt_index in range(gt_boxes_cnt):
            for pred_index in range(pred_boxes_cnt):
                if IOU(pred_boxes[pred_index], gt_boxes[gt_index]) >= threshold_iou:
                    recall_vector[gt_index] = True

                    if LABEL_DIC[pred_class_index[pred_index]] == gt_class_index[gt_index]:
                        precision_vector[pred_index] = True

        precision = np.sum(precision_vector) / pred_boxes_cnt
        recall = np.sum(recall_vector) / gt_boxes_cnt

        precision = precision.astype(np.float32)
        recall = recall.astype(np.float32)

    return precision, recall

def slice_tensor(x, start, end=None):
    if end is None:
        y = x[..., start:start+1]
    elif end < 0:
        y = x[..., start:]
    else:
        y = x[..., start:end+1]

    return y

def iou_wh(r1, r2):

	min_w = min(r1[0],r2[0])
	min_h = min(r1[1],r2[1])
	area_r1 = r1[0]*r1[1]
	area_r2 = r2[0]*r2[1]
		
	intersect = min_w * min_h		
	union = area_r1 + area_r2 - intersect

	return intersect/union

def get_grid_cell(roi, raw_w, raw_h, grid_w, grid_h):

	x_center = roi[0] + roi[2]/2.0
	y_center = roi[1] + roi[3]/2.0

	grid_x = int(x_center/float(raw_w)*float(grid_w))
	grid_y = int(y_center/float(raw_h)*float(grid_h))
		
	return grid_x, grid_y

def get_active_anchors(roi, anchors):
	 
	indxs = []
	iou_max, index_max = 0, 0
	for i,a in enumerate(anchors):
		iou = iou_wh(roi[2:], a)
		if iou>IOU_TH:
			indxs.append(i)
		if iou > iou_max:
			iou_max, index_max = iou, i

	if len(indxs) == 0:
		indxs.append(index_max)

	return indxs

def xml_read(xml_path, type = 'xywh'):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = xml_path[:-4] + '.jpg'
    image_path = image_path.replace('/xml', '/image')

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    bboxes = []
    classes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        classes.append(LABEL_DIC[label])

        bbox = obj.find('bndbox')
        
        bbox_xmin = int(bbox.find('xmin').text.split('.')[0])
        bbox_xmax = int(bbox.find('xmax').text.split('.')[0])
        bbox_ymin = int(bbox.find('ymin').text.split('.')[0])
        bbox_ymax = int(bbox.find('ymax').text.split('.')[0])

        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin

        if type == 'xywh':
            bboxes.append((bbox_xmin, bbox_ymin, bbox_width, bbox_height))
        else:
            bboxes.append((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

    return image_path, bboxes, classes

def read_xmls(xml_dir, xml_names):
    img_paths = []
    rois = []
    classes = []

    for xml_name in xml_names:
        img_path, _rois, _classes = xml_read(xml_dir + xml_name, 'xywh')

        img_paths.append(img_path)
        rois.append(_rois)
        classes.append(_classes)

    return img_paths, rois, classes

def roi2label(roi, anchor, raw_w, raw_h, grid_w, grid_h):
	
	x_center = roi[0]+roi[2]/2.0
	y_center = roi[1]+roi[3]/2.0

	grid_x = x_center/float(raw_w)*float(grid_w)
	grid_y = y_center/float(raw_h)*float(grid_h)
	
	grid_x_offset = grid_x - int(grid_x)
	grid_y_offset = grid_y - int(grid_y)

	roi_w_scale = roi[2]/float(raw_w)/anchor[0]
	roi_h_scale = roi[3]/float(raw_h)/anchor[1]

	label=[grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]
	
	return label

def onehot(idx, num):

	ret = np.zeros([num], dtype=np.float32)
	ret[idx] = 1.0

	return ret

def generate_yolov2_labels(xml_dirs):

    data = []

    for xml_dir in xml_dirs:

        xml_names = os.listdir(xml_dir)
        xml_img_paths, xml_rois, xml_classes = read_xmls(xml_dir, xml_names)

        for img_path, rois, classes in zip(xml_img_paths, xml_rois, xml_classes):

            img = cv2.imread(img_path)
            if img is None:
                continue

            raw_h, raw_w, _ = img.shape
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

            img = np.asarray(img, dtype = np.float32)
            label = np.zeros([GRID_H, GRID_W, N_ANCHORS, BOX_SIZE + 1], dtype = np.float32)
            
            for roi, cls in zip(rois, classes):
                active_idxs = get_active_anchors(roi, ANCHORS)
                grid_x, grid_y = get_grid_cell(roi, raw_w, raw_h, GRID_W, GRID_H)

                for active_idx in active_idxs:
                    anchor_label = roi2label(roi, ANCHORS[active_idx], raw_w, raw_h, GRID_W, GRID_H)
                    label[grid_y, grid_x, active_idx] = np.concatenate((anchor_label, [cls], [1.0]))
            
            data.append([img, label])
            
    return data

def get_generate_yolov2_labels(xml_dir, xml_names, normalize = True):

    data = []

    if True:
        xml_img_paths, xml_rois, xml_classes = read_xmls(xml_dir, xml_names)

        for img_path, rois, classes in zip(xml_img_paths, xml_rois, xml_classes):

            _img = cv2.imread(img_path)
            if _img is None:
                continue

            #if normalize:
            #    img = (_img - np.mean(_img)) / np.std(_img)
            #else:
            #    img = _img

            img = _img

            raw_h, raw_w, _ = img.shape
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

            img = np.asarray(img, dtype = np.float32)
            label = np.zeros([GRID_H, GRID_W, N_ANCHORS, BOX_SIZE + 1], dtype = np.float32)
            
            for roi, cls in zip(rois, classes):
                active_idxs = get_active_anchors(roi, ANCHORS)
                grid_x, grid_y = get_grid_cell(roi, raw_w, raw_h, GRID_W, GRID_H)

                for active_idx in active_idxs:
                    anchor_label = roi2label(roi, ANCHORS[active_idx], raw_w, raw_h, GRID_W, GRID_H)
                    label[grid_y, grid_x, active_idx] = np.concatenate((anchor_label, [cls], [1.0]))
            
            data.append([img, label])
            
    return data

def decode_yolov2_output(output, object_threshold = 0.5):

    objects = []

    for y in range(GRID_H):
        for x in range(GRID_W):
            for z in range(N_ANCHORS):
                anchor = ANCHORS[z]
                cx, cy, w, h, cls, conf = output[y, x, z, :]

                if conf >= object_threshold:
                    print(cx, cy, w, h, cls, conf)

                    w *= anchor[0]
                    h *= anchor[1]

                    w *= IMAGE_WIDTH
                    h *= IMAGE_HEIGHT

                    cx *= GRID_SIZE
                    cy *= GRID_SIZE

                    cx += (x * GRID_SIZE)
                    cy += (y * GRID_SIZE)

                    xmin = int((cx - w/2))
                    ymin = int((cy - h/2))
                    xmax = int((cx + w/2))
                    ymax = int((cy + h/2))

                    objects.append([[xmin, ymin, xmax, ymax], cls])

                    #print(CLASS_NAME[cls])
                    #print(xmin, ymin, xmax, ymax)
                    #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return objects

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def _decode_yolov2_output(output, object_threshold = 0.5):

    objects = []

    for y in range(GRID_H):
        for x in range(GRID_W):
            for z in range(N_ANCHORS):
                anchor = ANCHORS[z]

                cx, cy, w, h = output[y, x, z, :4]
                conf = sigmoid(output[y, x, z, 4])
                cls = output[y, x, z, 5:]

                if conf >= object_threshold:
                    cx = sigmoid(cx)
                    cy = sigmoid(cy)

                    w = np.exp(w)
                    h = np.exp(h)

                    w *= anchor[0]
                    h *= anchor[1]

                    w *= IMAGE_WIDTH
                    h *= IMAGE_HEIGHT

                    cx *= GRID_SIZE
                    cy *= GRID_SIZE

                    cx += (x * GRID_SIZE)
                    cy += (y * GRID_SIZE)

                    xmin = max(min(int((cx - w/2)), IMAGE_WIDTH - 1), 0)
                    ymin = max(min(int((cy - h/2)), IMAGE_HEIGHT - 1), 0)
                    xmax = max(min(int((cx + w/2)), IMAGE_WIDTH - 1), 0)
                    ymax = max(min(int((cy + h/2)), IMAGE_HEIGHT - 1), 0)

                    objects.append([[xmin, ymin, xmax, ymax], CLASS_NAME[np.argmax(cls)]])

                    #print(CLASS_NAME[cls])
                    #print(xmin, ymin, xmax, ymax)
                    #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return objects