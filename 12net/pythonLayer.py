import sys
# sys.path.append('/home/cmcc/caffe-master/python')
import cv2
import caffe
import numpy as np
import random
import cPickle as pickle
imdb_exit = True

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 384
        net_side = 12
        pos_list = ''
        neg_list = ''
        roi_list = ''
        landmark_list = ''
        pos_root = ''
        neg_root = ''
        roi_root = ''
        pts_root = ''
        print("Data_Layer_train")
        self.batch_loader = BatchLoader(pos_list, neg_list, roi_list, landmark_list, net_side, pos_root, neg_root, roi_root, pts_root)
        # image
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        # label
        top[1].reshape(self.batch_size, 1)
        # bbox
        top[2].reshape(self.batch_size, 4)
        # landmark
        top[3].reshape(self.batch_size, 10)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # loss_task = random.randint(0,2)
        # pos = 64, flag = 0
        # neg = 192, flag = 1
        # part = 64, flag = 2
        # landmark = 64, flag = 3

        self.batch_idxs = [0] * 64 + [1] * 192 + [2] * 64 + [3] * 64
        np.random.shuffle(self.batch_idxs)
        # print(self.batch_idxs)

        for idx, flag in enumerate(self.batch_idxs):
            # print(idx, flag)
            im, label, roi, pts= self.batch_loader.load_next_image(flag)
            # print(np.shape(im))
            # print(label)
            # print(roi)
            # print(pts)
            # print("_____________")
            top[0].data[idx, ...] = im
            top[1].data[idx, ...] = label
            top[2].data[idx, ...] = roi
            top[3].data[idx, ...] = pts

    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,pos_list, neg_list, part_list, landmark_list, net_side, pos_root, neg_root, part_root, landmark_root):
        self.mean = 128
        self.im_shape = net_side
        self.pos_root = pos_root
        self.neg_root = neg_root
        self.part_root = part_root
        self.landmark_root = landmark_root

        self.pos_list = []
        self.neg_list = []
        self.part_list = []
        self.landmark_list = []
        print "Start Reading Positive Data into Memory..."
        if imdb_exit:
            fid = open('12/positive.imdb','r')
            self.pos_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(pos_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.strip('\n').split(' ')
                image_file_name = self.pos_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                # im = np.swapaxes(im, 0, 2)
                im = np.transpose(im, (2,0,1))
                im -= self.mean
                label    = int(words[1])
                roi      = [-1,-1,-1,-1]
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.pos_list.append([im,label,roi,pts])
        random.shuffle(self.pos_list)
        self.pos_cur = 0
        print str(len(self.pos_list))," Positive Data have been read into Memory...", "\n"

        print "Start Reading Negative Data into Memory..."
        if imdb_exit:
            fid = open('12/negative.imdb','r')
            self.neg_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(neg_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.strip('\n').split(' ')
                image_file_name = self.neg_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                # im = np.swapaxes(im, 0, 2)
                im = np.transpose(im, (2,0,1))
                im -= self.mean
                label    = int(words[1])
                roi      = [-1,-1,-1,-1]
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.neg_list.append([im,label,roi,pts])
        random.shuffle(self.neg_list)
        self.neg_cur = 0
        print str(len(self.neg_list))," Negative Data have been read into Memory...", "\n"

        print "Start Reading Regression Data into Memory..."
        if imdb_exit:
            fid = open('12/part.imdb','r')
            self.part_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(part_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.strip('\n').split(' ')
                image_file_name = self.roi_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                # im = np.swapaxes(im, 0, 2)
                im = np.transpose(im, (2,0,1))
                im -= self.mean
                label    = int(words[1])
                roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.part_list.append([im,label,roi,pts])
        random.shuffle(self.part_list)
        self.part_cur = 0
        print str(len(self.part_list))," Regression Data have been read into Memory...", "\n"

        print "Start Reading Landmark-regression Data into Memory..."
        if imdb_exit:
            fid = open('12/landmark.imdb','r')
            self.landmark_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(landmark_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.pts_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                # im = np.swapaxes(im, 0, 2)
                im = np.transpose(im, (2,0,1))
                im -= self.mean
                label   = -1
                roi     = [-1,-1,-1,-1]
                pts     = [ float(words[ 6]),float(words[ 7]),
                            float(words[ 8]),float(words[ 9]),
                            float(words[10]),float(words[11]),
                            float(words[12]),float(words[13]),
                            float(words[14]),float(words[15])]
                self.landmark_list.append([im,label,roi,pts])
        random.shuffle(self.landmark_list)
        self.landmark_cur = 0
        print str(len(self.landmark_list))," Landmark-regression Data have been read into Memory...", "\n"

    def load_next_image(self, loss_task):
        if loss_task == 0:
            # get cls image, use positives
            if self.pos_cur == len(self.pos_list):
                    self.pos_cur = 0
                    random.shuffle(self.pos_list)
            cur_data = self.pos_list[self.pos_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            roi      = cur_data[2]
            pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            # random flip
            # 0 vertical flip 1 horizontal flip -1 mirror
            if random.choice([0,1])==1:
                im = np.transpose(im, (1,2,0))
                im = cv2.flip(im,random.choice([-1,0,1]))
                im = np.transpose(im, (2,0,1))
            self.pos_cur += 1
            return im, label, roi, pts

        if loss_task == 1:
            # get cls image, use negatives
            if self.neg_cur == len(self.neg_list):
                    self.neg_cur = 0
                    random.shuffle(self.neg_list)
            cur_data = self.neg_list[self.neg_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            roi      = [-1,-1,-1,-1]
            pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            # random flip
            # 0 vertical flip 1 horizontal flip -1 mirror
            if random.choice([0,1])==1:
                im = np.transpose(im, (1,2,0))
                im = cv2.flip(im,random.choice([-1,0,1]))
                im = np.transpose(im, (2,0,1))
            self.neg_cur += 1
            return im, label, roi, pts

        if loss_task == 2:
            # get bbox regression image, use(parts and positives)
            if self.part_cur == len(self.part_list):
                self.part_cur = 0
                random.shuffle(self.part_list)
            cur_data = self.part_list[self.part_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            roi      = cur_data[2]
            pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.part_cur += 1
            return im, label, roi, pts

        if loss_task == 3:
            # get landmark image, use(landmarks)
            if self.landmark_cur == len(self.landmark_list):
                self.landmark_cur = 0
                random.shuffle(self.landmark_list)
            cur_data = self.landmark_list[self.landmark_cur]  # Get the image index
            im	     = cur_data[0]
            label    = cur_data[1]
            roi      = [-1,-1,-1,-1]
            pts	     = cur_data[3]
            self.landmark_cur += 1
            return im, label, roi, pts


################################################################################
#########################ROI Loss Layer By Python###############################
################################################################################
class bbox_regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 3:
            raise Exception("Need 3 Inputs")
    def reshape(self,bottom,top):
        # print("bbox_regression_Layer")
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")
        roi = bottom[1].data
        label = bottom[2].data
        # use positive and part to do bbox regression
        # positive label == 1, fixed num 64
        # part label == -1, fixed num 64

        # get positive idxs
        # print(type(np.where(label[:] == 1)[0]))
        self.valid_index = list(np.where(label[:] == 1)[0]) + list(np.where(label[:] == -1)[0])
        # print(len(self.valid_index))
        # print(self.valid_index)

        self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
        self.diff[...] = 0
        top[0].data[...] = 0
        if self.N != 0:
            # bottom[0] is pred & bottom[1] is roi & bottom[2] is label
            # print(bottom[0].data[self.valid_index])
            # print(bottom[1].data[self.valid_index])
            self.diff[self.valid_index] = bottom[0].data[self.valid_index] - np.array(bottom[1].data[self.valid_index]).reshape(bottom[0].data[self.valid_index].shape)
            # divide 2, because loss weight is 0.5
            top[0].data[...] = np.sum(self.diff**2) / self.N / 2.

    def backward(self,top,propagate_down,bottom):
        for i in range(2):
            if not propagate_down[i] or self.N==0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            # top[0].diff[0] is loss weight
            # print(top[0].diff[0])
            bottom[i].diff[self.valid_index] = sign * top[0].diff[0] * self.diff[self.valid_index] / self.N

class landmark_regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 3:
            raise Exception("Need 3 Inputs")
    def reshape(self,bottom,top):
        # print("landmark_regression_Layer")
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")
        roi = bottom[1].data
        label = bottom[2].data
        # use landmark to do landmark regression
        # landmark label == -2
        # part label == -1
        # self.valid_index = np.where((label[:] == 1 or label[:] == -1))[0]
        # print(self.valid_index)
        self.valid_index = list(np.where(label[:] == -2)[0])
        # print(len(self.valid_index))
        # print(self.valid_index)

        self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
        self.diff[...] = 0
        top[0].data[...] = 0
        if self.N != 0:
            # bottom[0] is pred & bottom[1] is roi & bottom[2] is label
            # print(bottom[0].data[self.valid_index])
            # print(bottom[1].data[self.valid_index])
            self.diff[self.valid_index] = bottom[0].data[self.valid_index] - np.array(bottom[1].data[self.valid_index]).reshape(bottom[0].data[self.valid_index].shape)
            # divide 2, because loss weight is 0.5
            top[0].data[...] = np.sum(self.diff**2) / self.N / 2.

    def backward(self,top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i] or self.N==0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            # print(top[0].diff[0])
            bottom[i].diff[self.valid_index] = sign * top[0].diff[0] * self.diff[self.valid_index] / self.N
################################################################################
#############################SendData Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
        # print("cls_Layer_fc")
        label = bottom[1].data
        # use postives and negatives to do regression
        # positive label == 1
        # negative label == 0
        self.valid_index = list(np.where(label[:] == 1)[0]) + list(np.where(label[:] == 0)[0])
        # print(len(self.valid_index))
        # print(self.valid_index)

        self.N = len(self.valid_index)
        # face & non-face
        top[0].reshape(self.N, 2,1,1)
        # label
        top[1].reshape(self.N, 1)

    def forward(self,bottom,top):
        top[0].data[...][...] = 0
        top[1].data[...][...] = 0
        # cls pred of positives and negatives
        top[0].data[...] = bottom[0].data[self.valid_index]
        # labels of positives and negatives
        top[1].data[...] = bottom[1].data[self.valid_index]

    def backward(self,top,propagate_down,bottom):
        if propagate_down[0] and self.N != 0:
            bottom[0].diff[...] = 0
            bottom[0].diff[self.valid_index] = top[0].diff[...]
        if propagate_down[1] and self.N != 0:
            bottom[1].diff[...] = 0
            bottom[1].diff[self.valid_index] = top[1].diff[...]
