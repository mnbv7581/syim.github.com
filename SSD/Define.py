
import numpy as np

ROOT_PATH = "D:/_ImageDataset/VOC2007,2012/VOC2007_Train/"
IMG_PATH = ROOT_PATH + "JPEGImages/"
XML_PATH = ROOT_PATH + "Annotations/"

#VOC
CLASS_NAME = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
LABELS = {k: v for v, k in enumerate(CLASS_NAME)}

CLASSES = len(LABELS.keys())
BACKGROUND_CLASS = 0

BATCH_SIZE = 8
MAX_EPOCHS = 100000

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CHANNEL = 3

#SSD Parameters

DEFAULT_BOX_SIZE = [6, 6, 6, 6, 6, 6]
DEFAULT_BOX_SCALE = np.linspace(0.1, 0.9, num=np.amax(DEFAULT_BOX_SIZE))
DEFAULT_BOX_ASPECT_RATIO = [[0.5, 1.0, 2.0, 3.0, 1.0 / 3.0]] * 6

print(DEFAULT_BOX_SCALE)

JACCARD_VALUE = 0.50

#feature_map_shape = []
#all_default_boxes = []

