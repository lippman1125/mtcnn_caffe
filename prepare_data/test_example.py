import sys
import copy
# sys.path.append('../demo/')
# sys.path.append('.')
# sys.path.append('/home/cmcc/caffe-master/python')
# import tools
import caffe
import cv2
import numpy as np
import os
import MtcnnDetector

from utils import *
deploy = '../12net/12net.prototxt'
caffemodel = '../12net/PNet_iter_1200000.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '../24net/24net.prototxt'
caffemodel = '../24net/RNet_iter_1200000.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '../48net/48net.prototxt'
caffemodel = '../48net/ONet_iter_1200000.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()

image_size = 48

anno_file = '/home/lqy/D/Projects/dream1992.txt'
im_dir = "/home/lqy/D/Projects/"

threshold = [0.7,0.7,0.9]
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
image_idx = 0

detectors = [None, None, None]
# use pnet
detectors[0] = net_12
detectors[1] = net_24
detectors[2] = net_48

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    # bbox = map(float, annotation[1:])
    # gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img_path = im_dir + annotation[0] + '.jpg'
    # rectangles = detectFace(img_path,threshold)
    img = cv2.imread(img_path)
    mtcnn_detector = MtcnnDetector.MtcnnDetector(detectors=detectors, min_face_size=20,
                                   stride=2, threshold=threshold, slide_window=False)
    rectangles, landmarks = mtcnn_detector.detect_single_image(img)
    # dets = convert_to_square(rectangles)
    dets = rectangles
    # for debug
    print(np.shape(dets))

    for i in range(len(dets)):
        x_left, y_top, x_right, y_bottom, _ = dets[i]
        # if x_right - x_left < 100 or y_bottom - y_top < 100 or x_right - x_left > 160 or y_bottom - y_top > 160:
        #     continue
        print(x_left, y_top, x_right, y_bottom, _)
        cv2.rectangle(img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0,0,255), 2)

        # pts = landmarks[i]
        # print(pts)
        # cv2.circle(img, (int(pts[0]), int(pts[1])), 3, (0, 255, 0), 2)
        # cv2.circle(img, (int(pts[2]), int(pts[3])), 3, (0, 255, 0), 2)
        # cv2.circle(img, (int(pts[4]), int(pts[5])), 3, (0, 255, 0), 2)
        # cv2.circle(img, (int(pts[6]), int(pts[7])), 3, (0, 255, 0), 2)
        # cv2.circle(img, (int(pts[8]), int(pts[9])), 3, (0, 255, 0), 2)

    cv2.imshow("{}".format(annotation[0]), img)
    cv2.imwrite(os.path.join(im_dir, annotation[0] + "_detection.jpg"), img)
    cv2.waitKey(0)
