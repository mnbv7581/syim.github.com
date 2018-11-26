
DB_XML_DIRS = ['../../DB/VOC/xml/']
DB_IMG_DIRS = ['../../DB/VOC/image/']
DB_XML_DIRS2 = ['../../DB/VOC/xml2/']
DB_IMG_DIRS2 = ['../../DB/VOC/image2/']
#VOC
CLASS_NAME = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#KEY : VALUE (0 : 'aeroplane')
LABEL_DIC = {k: v for v, k in enumerate(CLASS_NAME)}

IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
IMAGE_CHANNEL = 3

filters = [32, 64, 128, 256]

GRID_W = 13
GRID_H = 13

GRID_SIZE = int(IMAGE_WIDTH / GRID_W)

BOX_SIZE = 5
N_CLASSES = len(CLASS_NAME)

IOU_TH = 0.5

N_ANCHORS = 5
ANCHORS = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

TENSORBOARD_LOGDIR = './logs/'

COORD = 5.0
NOOBJ = 0.5

BATCH_SIZE= 16

N_ITERS = 300000
LOG_ITER = 100
SAVE_ITER = 1000