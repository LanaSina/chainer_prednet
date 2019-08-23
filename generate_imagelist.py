import argparse
import numpy as np
import os
#import sys


usage = 'Usage: python {} DATA_DIR [N_IMAGES] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate text list files image datasets',
                                 usage=usage)
parser.add_argument('data_dir', action='store', nargs=None, 
                    type=str, help='path to directory containing the input_images _folder_.')
parser.add_argument('n_images', action='store', nargs='?', default=-1,
                    type=int, help='optional: total number of images to use.')
args = parser.parse_args()

split_ratio = np.array([0,1])
splits = ["train", "test"]

n_images = args.n_images
data_dir = args.data_dir
input_images_dir = "input_images/"
im_dir = data_dir + "/" + input_images_dir
image_list = sorted(os.listdir(im_dir))

if n_images==-1:
    n_images = len(image_list)
else:
	image_list = image_list[:n_images]

limits = split_ratio*n_images
print(n_images, " images")
print(limits)

train_list_file = os.path.join(data_dir, "train_list.txt")
test_list_file = os.path.join(data_dir, "test_list.txt")

print('\nSave %s' % train_list_file)
with open(train_list_file, 'w') as f:
    f.write(input_images_dir)
    tmp = "\n" + input_images_dir
    f.write(tmp.join(image_list[:int(limits[0])]))
print('Save %s' % test_list_file)
with open(test_list_file, 'w') as f:
    f.write(input_images_dir)
    tmp = "\n" + input_images_dir
    f.write(tmp.join(image_list[int(limits[0]):]))
print("Done.")
