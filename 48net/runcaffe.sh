#!/usr/bin/env sh
export PYTHONPATH=/home/lqy3/caffe/python:/home/lqy3/mtcnn-caffe/48net

set -e
/home/lqy3/caffe/build/tools/caffe train \
	 --solver=./solver.prototxt --gpu 0\
