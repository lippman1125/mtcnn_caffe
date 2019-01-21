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
# caffemodel = "../12net/det1.caffemodel"
caffemodel = '../12net/PNet_iter_1200000.caffemodel'
# caffemodel = '../12net/12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '../24net/24net.prototxt'
# caffemodel = "../24net/24net.caffemodel"
caffemodel = '../24net/RNet_iter_360000.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()
# def detectFace(img_path,threshold):
#     img = cv2.imread(img_path)
#     # caffe_img = img.copy()-128
#     origin_h,origin_w,ch = img.shape
#     scales = tools.calculateScales(img)
#     out = []
#     for scale in scales:
#         hs = int(origin_h*scale)
#         ws = int(origin_w*scale)
#         scale_img = cv2.resize(img,(ws,hs))
#         scale_img = np.swapaxes(scale_img, 0, 2)
#         scale_img = (scale_img - 127.5)/128
#         print(scale_img)
#         net_12.blobs['data'].reshape(1,3,ws,hs)
#         net_12.blobs['data'].data[...]=scale_img
#         caffe.set_device(0)
#         caffe.set_mode_gpu()
#         out_ = net_12.forward()
#         out.append(out_)
#     image_num = len(scales)
#     rectangles = []
#     for i in range(image_num):
#         cls_prob = out[i]['prob1'][0][1]
#         roi      = out[i]['conv4-2'][0]
#         out_h,out_w = cls_prob.shape
#         out_side = max(out_h,out_w)
#         rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
#         rectangles.extend(rectangle)
#     return rectangles

image_size = 48

anno_file = 'wider_face_train.txt'
im_dir = "WIDER_train/images/"
neg_save_dir  = "../{}net/{}/negative".format(image_size, image_size)
pos_save_dir  = "../{}net/{}/positive".format(image_size, image_size)
part_save_dir = "../{}net/{}/part".format(image_size, image_size)

if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)
if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.makedirs(part_save_dir)

ensure_directory_exists(neg_save_dir)
ensure_directory_exists(pos_save_dir)
ensure_directory_exists(part_save_dir)

f1 = open('../{}net/{}/pos_{}.txt'.format(image_size, image_size, image_size), 'w')
f2 = open('../{}net/{}/neg_{}.txt'.format(image_size, image_size, image_size), 'w')
f3 = open('../{}net/{}/part_{}.txt'.format(image_size, image_size, image_size), 'w')
threshold = [0.6,0.6,0.7]
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

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    bbox = map(float, annotation[1:])
    gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img_path = im_dir + annotation[0] + '.jpg'
    # rectangles = detectFace(img_path,threshold)
    img = cv2.imread(img_path)
    mtcnn_detector = MtcnnDetector.MtcnnDetector(detectors=detectors, min_face_size=20,
                                   stride=2, threshold=threshold, slide_window=False)
    rectangles, _ = mtcnn_detector.detect_single_image(img)
    if rectangles is None:
        print("{} can't found faces".format(img_path))
        continue
    dets = convert_to_square(rectangles)
    # for debug
    # print(np.shape(dets))
    # for box in dets:
    #     x_left, y_top, x_right, y_bottom, _ = box
    #     # if x_right - x_left < 100 or y_bottom - y_top < 100 or x_right - x_left > 160 or y_bottom - y_top > 160:
    #     #     continue
    #     cv2.rectangle(img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0,0,255), 2)
    #
    # cv2.imshow("pnet_detect", img)
    # cv2.waitKey(0)
    # exit()

    image_idx += 1
    view_bar(image_idx,num)
    neg_num = 0
    # print(gts)
    for box in dets:
        x_left, y_top, x_right, y_bottom, _ = box

        # convert float to int
        x_left = int(x_left)
        y_top  = int(y_top)
        x_right = int(x_right)
        y_bottom = int(y_bottom)

        crop_w = x_right - x_left + 1
        crop_h = y_bottom - y_top + 1

        # ignore box that is too small or beyond image border
        if crop_w < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
            continue

        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, gts)
        cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        # print(Iou)
        # save negative images and write label
        if np.max(Iou) < 0.3 and neg_num < 100:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("../%snet/%s/negative/%s"%(image_size, image_size, n_idx) + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
        else:
            # find gt_box with the highest iou
            idx = np.argmax(Iou)
            assigned_gt = gts[idx]
            x1, y1, x2, y2 = assigned_gt

            # compute bbox reg label
            offset_x1 = (x1 - x_left)   / float(crop_w)
            offset_y1 = (y1 - y_top)    / float(crop_h)
            offset_x2 = (x2 - x_right)  / float(crop_w)
            offset_y2 = (y2 - y_bottom )/ float(crop_h)

            # save positive and part-face images and write labels
            if np.max(Iou) >= 0.65:
                # print(box)
                # print("______________")
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("../%snet/%s/positive/%s"%(image_size, image_size, p_idx) + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

            elif np.max(Iou) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("../%snet/%s/part/%s"%(image_size, image_size, d_idx)     + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
    # exit()

f1.close()
f2.close()
f3.close()
