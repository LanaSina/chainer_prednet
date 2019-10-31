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
  h = 120

  fieldnames = ['r','g','b']

  with open(output_dir+output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)

    for image_file in image_list[0:image_count]:
      image_path = os.path.join(image_dir, image_file)
      print("read ", image_path)

      # h, w, color
      current_image = numpy.array(Image.open(image_path).convert('RGB'))
      r = 0.0
      g = 0.0
      b = 0.0

      for x in range(80-55,80+55):
      	for y in range(60-15,60+15):
      		r = r + current_image[y, x, 0]
      		g = g + current_image[y, x, 1]
      		b = b + current_image[y, x, 2]

      total = (55*2)*(15*2)
      r = r / total
      g = g / total
      b = b / total
      row = [r,g,b]
      writer.writerow(row)

def row_averages(image_dir, output_dir, n, start, output_file):
  image_list = sorted(os.listdir(image_dir))
  image_count =  n

  if(n == -1):
    image_count =  len(image_list)

  w = 160
  h = 120

  fieldnames = ['r','g','b']

  with open(output_dir+output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)

    for image_file in image_list[start:start+image_count]:
      image_path = os.path.join(image_dir, image_file)
      print("read ", image_path)

      # h, w, color
      current_image = numpy.array(Image.open(image_path).convert('RGB'))
      r = numpy.zeros(h)
      g = numpy.zeros(h)
      b = numpy.zeros(h)

      for x in range(0,w):
      	for y in range(0,h):
      		r[y] = r[y] + current_image[y, x, 0]
      		g[y] = g[y] + current_image[y, x, 1]
      		b[y] = b[y] + current_image[y, x, 2]

      r = r / w
      g = g / w
      b = b / w
      rows = numpy.array([r,g,b])
      rows = rows.transpose()
      rows_list = rows.tolist()
      writer.writerows(rows)


parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--image_dir', '-i', default='', help='Path to images folder')
parser.add_argument('--n_images', '-n', default=-1, type=int, help='number of images to process')
parser.add_argument('--output_file', '-o', default='', help='name of output csv')
parser.add_argument('--start', '-s', default=0, type=int, help='start')



args = parser.parse_args()

row_averages(args.image_dir, output_dir, args.n_images, args.start, args.output_file)
