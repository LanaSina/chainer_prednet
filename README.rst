#coding:utf-8


================================
PredNet in chainer
================================
Kenta Tanaka & Eiji Watanabe, 2019



================================
Overview
================================
This software learns a still image sequence generated from a video by a deep learning algorithm, and generates predicted video.



================================
Test environment
================================
OS: Ubuntu 16.04
Python: 2.7
GPU: Nvidia GTX1080ti
chainer==5.30



================================
Requirements
================================
* python-opencv
* g++
* python-dev
* libhdf5-dev
* PIL
* protobuf
* filelock
* cython
* setuptools
* chainer
* numpy
* cupy
* net
* cv2
* os
* sys
* argparse
* datetime
* setuptools
* tensorflow
* tensorboard-chainer
* 
* GPU driver, CUDA Toolkit and cudnn



================================
Preparing data
================================
Put the target movie file (mp4) in "data" folder.
Execute the following command to generate still images from the movie.

$ python generate_image.py data/YOUR_VIDEO -d data

To change the width of the image, use the -w option.

$ python generate_image.py data/YOUR_VIDEO -d data -w 160

The height can also be specified with -g option.

$ python generate_image.py data/YOUR_VIDEO -d data -w 160 -g 120

If no option is specified, the width is fixed to 160 and the height is fixed to 120.
Also, "train_list.txt" describing the list of files used for training
and "test_list.txt" describing the list of files used for testing are saved.
By default, the latter half of the video will be the test data.

$ python generate_image.py data/YOUR_VIDEO -d data -s 100
You can specify the number of frames to skip with -s option.

$ python generate_image.py data/YOUR_VIDEO -d data -n 50
With -n option, you can copy n frames of the same frame.



================================
Training
================================
Next, execute the following command for training.

$ python PredNet/main.py -i data/train_list.txt

To use GPU, execute as follows.

$ python PredNet/main.py -i data/train_list.txt -g 0

The learning models are saved in "models folder"
 and the generated images are saved "image folder".



================================
Prediction
================================
Next, generate predicted frames with the following command.

$ python PredNet/main.py -i data/test_list.txt --test --initmodel models/YOUR_MODEL -l NUMBER_OF_INPUT_IMAGES --ext NUMBER_OF_PREDICTED_IMAGES

Predicted images (test_#y_ 0.jpg) of all the images described in "test_list.txt" are generated in "result folder".
Furthermore, for each length of the input image, images (test_#y_1.jpg, test_#y_2.jpg, ...) corresponding to the number of predicted frames are generated.



================================
Options
================================
parser.add_argument('--images', '-i', default='', help='Path to image list file')
parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.', help='Root directory path of sequence and image files')
parser.add_argument('--initmodel', default='', help='Initialize the model from given file')
parser.add_argument('--resume', default='', help='Resume the optimization from snapshot')
parser.add_argument('--size', '-s', default='160,120', help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192', help='Number of channels on each layers')
parser.add_argument('--offset', '-o', default='0,0', help='Center offset of clipping input image (pixels)')
parser.add_argument('--input_len', '-l', default=50, type=int, help='Input frame length fo extended prediction on test (frames)')
parser.add_argument('--ext', '-e', default=10, type=int, help='Extended prediction on test (frames)')
parser.add_argument('--bprop', default=20, type=int, help='Back propagation length (frames)')
parser.add_argument('--save', default=10000, type=int, help='Period of save model and state (frames)')
parser.add_argument('--period', default=1000000, type=int, help='Period of training (frames)')



================================
How to Use Tensorboard

https://github.com/neka-nat/tensorboard-chainer
================================
To use tensorboard, install "tensorboard-chainer" by execute the following command.

$ sudo pip install tensorflow

then
$ sudo pip install tensorboard-chainer


To update "tensorboard-chainer",execute the following command.

$ sudo pip install -U tensorboard-chainer


To use tensorboard-chainer, Execute the following command,

$ tensorboard --logdir runs

and access "http://localhost:6006" in your browser.



================================
Reference
================================
"https://github.com/neka-nat/" [Powered by Tanaka]

"https://coxlab.github.io/prednet/" [Original PredNet]
"https://github.com/quadjr/PredNet" [Implemented by chainer]



