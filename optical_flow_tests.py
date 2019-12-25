import argparse
import os
#import sys
#sys.path.insert(1, '/PredNet')
#import main as prednet_main
from PredNet.call_prednet import call_prednet
from generate_img_func import generate_imagelist



# process images as static images
def predict_static(input_path, model_name, limit):
	# how many times to repeat each image
	repeat = 10

	print("create image list")
	# 'chainer_prednet/generate_imagelist.py' 'imported/' '1' -tr 10
	imagelist_args = {"data_dir": input_path, "n_images": limit, "r": repeat}
	generate_imagelist(imagelist_args)

	print("run prednet")
	# only save last image
	# %run 'chainer_prednet/PredNet/main.py' --images 'imported' --initmodel 'fpsi_500000_20v.model' --input_len 10 --test 
	l = limit*10
	prednet_args = {"images":input_path, "initmodel":model_name, "input_len":l, "test":0, "skip":repeat}
	call_prednet(prednet_args)

	# calculate optical flow compared to input


# do the mirror test


parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to input directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--limit', '-l', type=int, default=0, help='max number of images')


args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

predict_static(args.input, args.model, args.limit)
