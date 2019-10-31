import argparse
import csv
import cv2
import numpy
import os
from PIL import Image
import random



output_dir = "image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def averages(image_dir, output_dir, n, output_file):
  image_list = sorted(os.listdir(image_dir))
  image_count =  n

  if(n == -1):
    image_count =  len(image_list)

  w = 160
  h = 90

  fieldnames = ['r','g','b']

  


parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--image_dir', '-i', default='', help='Path to images folder')
parser.add_argument('--n_images', '-n', default=-1, type=int, help='number of images to process')
parser.add_argument('--output_file', '-o', default='', help='name of output csv')


args = parser.parse_args()

averages(args.image_dir, output_dir, args.n_images, args.image_dir)
