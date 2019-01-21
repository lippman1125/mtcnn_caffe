import numpy as np
import numpy.random as npr
import sys
import cv2
import os
import numpy as np
import cPickle as pickle

size = 48
net = '../{}net/'.format(size) + str(size)

positive_imdb = os.path.join(net, "positive.imdb")
negative_imdb = os.path.join(net, "negative.imdb")
part_imdb     = os.path.join(net, "part.imdb")
landmark_imdb = os.path.join(net, "landmark.imdb")

with open('%s/pos_%s.txt'%(net, size), 'r') as f:
    pos = f.readlines()

with open('%s/neg_%s.txt'%(net, size), 'r') as f:
    neg = f.readlines()

with open('%s/part_%s.txt'%(net, size), 'r') as f:
    part = f.readlines()

with open('%s/landmark_%s.txt'%(net, size), 'r') as f:
    landmark = f.readlines()
    
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
    

positive_list = []
print('\n'+ 'positive-{}'.format(size))
cur_ = 0
sum_ = len(pos)
for line in pos:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.strip('\n').split(' ')
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    if im is None:
        print("Pos {} is None".format(image_file_name))
        continue
    h,w,ch = im.shape
    if h != size or w != size:
        im = cv2.resize(im,(size, size))
    # im = np.swapaxes(im, 0, 2)
    im = np.transpose(im, (2,0,1))
    im = (im - 127.5)/128
    label    = 1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    positive_list.append([im, label, roi, pts])
fid = open(positive_imdb, 'w')
pickle.dump(positive_list, fid)
fid.close()


nagative_list = []
print ('\n'+ 'negative-{}'.format(size))
cur_ = 0
neg_keep = npr.choice(len(neg), size=300000, replace=False)
sum_ = len(neg_keep)
for i in neg_keep:
    line = neg[i]
    view_bar(cur_, sum_)
    cur_ += 1
    words = line.strip('\n').split(' ')
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    if im is None:
        print("Neg {} is None".format(image_file_name))
        continue
    h,w,ch = im.shape
    if h != size or w != size:
        im = cv2.resize(im,(size, size))
    # im = np.swapaxes(im, 0, 2)
    im = np.transpose(im, (2,0,1))
    im = (im - 127.5)/128
    label    = 0
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    nagative_list.append([im, label, roi, pts])
fid = open(negative_imdb, 'w')
pickle.dump(nagative_list, fid)
fid.close()


part_list = []
print '\n'+'part-{}'.format(size)
cur_ = 0
# part_keep = npr.choice(len(part), size=500000, replace=False)
sum_ = len(part)
for line in part:
    view_bar(cur_, sum_)
    cur_ += 1
    words = line.strip('\n').split(' ')
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    if im is None:
        print("Neg {} is None".format(image_file_name))
        continue
    h,w,ch = im.shape
    if h != size or w != size:
        im = cv2.resize(im,(size, size))
    # im = np.swapaxes(im, 0, 2)
    im = np.transpose(im, (2,0,1))
    im = (im - 127.5) / 128
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    part_list.append([im, label, roi, pts])
fid = open(part_imdb, 'w')
pickle.dump(part_list, fid)
fid.close()

landmark_list = []
print('\n'+'landmark-{}'.format(size))
cur_ = 0
sum_ = len(landmark)
for line in landmark:
    view_bar(cur_, sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    if im is None:
        print("Neg {} is None".format(image_file_name))
        continue
    h,w,ch = im.shape
    if h != size or w != size:
        im = cv2.resize(im,(size, size))
    # im = np.swapaxes(im, 0, 2)
    im = np.transpose(im, (2,0,1))
    im = (im - 127.5)/128
    label    = -2
    roi      = [-1, -1, -1, -1]
    pts	     = [float(words[2]),float(words[3]),
                float(words[4]),float(words[5]),
                float(words[6]),float(words[7]),
                float(words[8]),float(words[9]),
                float(words[10]),float(words[11])]
    landmark_list.append([im, label, roi, pts])

fid = open(landmark_imdb,'w')
pickle.dump(landmark_list, fid)
fid.close()