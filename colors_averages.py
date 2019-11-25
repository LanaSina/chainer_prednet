import argparse
import csv
import cv2
import numpy
import os
from PIL import Image
import random


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


def singular_rows(image_path, output_dir, output_file, x_start, y_start, w, h):

  fieldnames = ['lum']
  save_file = output_dir + "/" + output_file
  print("save in " + save_file)

  with open(save_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)

    print("read ", image_path)

    # h, w, color
    current_image = numpy.array(Image.open(image_path).convert('RGB'))
    
    #lum = numpy.zeros(h*w)
    # g = numpy.zeros(h)
    # b = numpy.zeros(h)


    for yy in range(0,h):
      y = yy + y_start
      for xx in range(0,w):
        x = xx + x_start
        lum = numpy.mean(current_image[y, x])
        row = [lum]
        writer.writerow(row)

    # r = r / w
    # g = g / w
    # b = b / w
    # rows = numpy.array([r,g,b])
    # rows = rows.transpose()
    # rows_list = rows.tolist()
    # writer.writerows(rows)


def row_averages(image_dir, output_dir, n, start, output_file, x_start, x_end):
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

      for x in range(x_start,x_end):
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
parser.add_argument('--output_dir', '-d', default='', help='path of output diectory')
parser.add_argument('--output_file', '-o', default='', help='name of output csv')
parser.add_argument('--x_start', '-xs', default=0, type=int, help='x start')
parser.add_argument('--y_start', '-ys', default=0, type=int, help='y start')
parser.add_argument('--width', '-w', default=0, type=int, help='width')
parser.add_argument('--height', '-he', default=0, type=int, help='height')



args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

w = 160
#row_averages(args.image_dir, output_dir, args.n_images, args.start, args.output_file, 0, w)
singular_rows(args.image_dir, output_dir, args.output_file, args.x_start, args.y_start, args.width, args.height)

# row_averages(args.image_dir, output_dir, args.n_images, args.start, "left_" + args.output_file, 0, int(w/2))
# row_averages(args.image_dir, output_dir, args.n_images, args.start, "right_" + args.output_file, int(w/2), w)
