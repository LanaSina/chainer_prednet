import argparse
import os
from generate_img_func import generate_imagelist

usage = 'Usage: python {} DATA_DIR [N_IMAGES] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate text list files image datasets',
                                 usage=usage)
parser.add_argument('data_dir', action='store', nargs=None, 
                    type=str, help='path to directory containing the input_images _folder_.')
parser.add_argument('n_images', action='store', nargs='?', default=-1,
                    type=int, help='optional: total number of images to use.')
parser.add_argument('--rep', '-r', type=int, default=1, help='number of times to repeat each image in test set')
parser.add_argument('--total_rep', '-tr', type=int, default=1, help='number of times to repeat the entire dataset')
args = parser.parse_args()

generate_imagelist(args)