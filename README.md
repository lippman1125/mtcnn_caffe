# mtcnn-caffe
Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks.

This project provide you a method to update multi-task-loss for multi-input source.


## Requirement
0. Ubuntu 16.04
1. caffe && pycaffe: [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)
2. cPickle && cv2 && numpy 

## Train Data
The training data generate process can refer to [AITTSMD/MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)

Sample almost similar to Seanlinx's can be found in `prepare_data`

- step1. Download Wider Face Training part only from Official Website and unzip to replace `WIDER_train`

- step2. Run `gen_12net_data.py`   and `gen_12net_landmark.py`to generate 12net training data. Run `gen_12net_imdb.py` to generate imdb.

- step3. Run `gen_rnet_example.py` and `gen_24net_landmark.py`to generate 24net training data. Run `gen_24net_data.py` to generate imdb.

- step4. Run `gen_onet_example.py` and `gen_48net_landmark.py`to generate 48net training data. Run `gen_48net_data.py` to generate imdb. 

- Note:<br>
       
       1. pos:neg:part:landmark == 64:192:64:64, so the batchsize = 384.
       2. 12net training. we use all pos pics, 100w neg pics, 100w part pics, all landmark pics.
       3. 24net training. we use all pos pics, 60w  neg pics, 50w  part pics, all landmark pics.
       4. 48net training. we use all pos pics. 30w  neg pics, all  part pics, all landmark pics.
       5. why we do this? because we load the data into memory once, so hi-res inputs need too much memory, we reduce number of them.


## Eval
![dream1992](https://github.com/lippman1125/github_images/blob/master/mtcnn_images/dream1992_detection.jpg)<br>
![office](https://github.com/lippman1125/github_images/blob/master/mtcnn_images/office_detection.jpg)<br>
![picture](https://github.com/lippman1125/github_images/blob/master/mtcnn_images/picture_detection.jpg)

## Reference
0. https://github.com/AITTSMD/MTCNN-Tensorflow
1. https://github.com/CongWeilin/mtcnn-caffe