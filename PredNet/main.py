import argparse
import os
from call_prednet import call_prednet


parser = argparse.ArgumentParser(
description='PredNet')
parser.add_argument('--images_path', '-i', default='', help='Path input images')
parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of sequence and image files')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels)')
parser.add_argument('--input_len', '-l', default=50, type=int,
                    help='Input frame length fo extended prediction on test (frames)')
parser.add_argument('--ext', '-e', default=0, type=int,
                    help='Extended prediction on test (frames)')
parser.add_argument('--ext_t', default=20, type=int,
                    help='When to start extended prediction')
parser.add_argument('--bprop', default=20, type=int,
                    help='Back propagation length (frames)')
parser.add_argument('--save', default=10000, type=int,
                    help='Period of save model and state (frames)')
parser.add_argument('--period', default=1000000, type=int,
                    help='Period of training (frames)')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--skip_save_frames', '-sikp', type=int, default=1, help='predictions will be saved every x steps')

parser.set_defaults(test=False)
args = parser.parse_args()

call_prednet(args)