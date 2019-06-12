#!/usr/bin/python
#coding:utf-8

import cv2
import os
import sys
import argparse

usage = 'Usage: python {} INPUT_FILE [--prefix <prefix>] [--dir <directory>] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate images from a video.',
                                 usage=usage)
parser.add_argument('input_video', action='store', nargs=None, 
                    type=str, help='Input video.')
parser.add_argument('-p', '--prefix', action='store', nargs='?',
                    default='frame', type=str, help='Prefix of Output file name.')
parser.add_argument('-d', '--dir', action='store', nargs='?',
                    default='data', type=str, help='Directory of Output files.')
parser.add_argument('-r', '--ratio', action='store',
                    default=0.1, type=float, help='Ratio of test datum.')
parser.add_argument('-w', '--width', action='store',
                    default=-1, type=int, help='Width of images.')
parser.add_argument('-g', '--height', action='store',
                    default=-1, type=int, help='height of images.')
parser.add_argument('-n', '--n_copy', action='store',
                    default=1, type=int, help='Number to copy images.')
parser.add_argument('-s', '--skip', action='store',
                    default=1, type=int, help='Skip amount of data.')
parser.add_argument('-f', '--flip', action='store_true',
                    default=False, help='Flip image.')
                    
args = parser.parse_args()

vidcap = cv2.VideoCapture(args.input_video)
success, image = vidcap.read()
count = 0
cur_skip = 0
files = []

print("Start to save images...")

while True:
    success, image = vidcap.read()
    if not success:
        break
    if cur_skip != args.skip:
        count += 1
        cur_skip += 1
        continue
    cur_skip = 0
    if args.flip:
        image = cv2.flip(image, 1)
    for n in range(args.n_copy):
        files.append(os.path.join(args.dir, "%s_%05d_%03d.jpg" % (args.prefix, count, n)))
        sys.stdout.write('\rSave {}'.format(files[-1]))
        sys.stdout.flush()
        if args.width > 0:
            height, width = image.shape[0], image.shape[1]
            if args.height < 0:
                height = int(height * float(args.width) / width)
            else:
                height = args.height
            image = cv2.resize(image, (args.width, height))
        cv2.imwrite(files[-1], image)
    count += 1
    cur_skip += 1


train_list_file = os.path.join(args.dir, "train_list.txt")
test_list_file = os.path.join(args.dir, "test_list.txt")
ratio = max(0.0, min(1.0, args.ratio))
index = int(count * (1.0 - ratio)) * args.n_copy

print('\nSave %s' % train_list_file)

with open(train_list_file, 'w') as f:
    f.write('\n'.join(files[:index]))
    
print('Save %s' % test_list_file)

with open(test_list_file, 'w') as f:
    f.write('\n'.join(files[index:]))
    
print("Done.")
