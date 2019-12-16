import argparse
import copy
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter

# cuts the images into w*h 5*4 parts
# copies the 1:1 space into all slots (avoiding black lines at the top of image and boundary conditions)
def copy(input_path, output_dir, copy_x, copy_y, limit):

  input_list = sorted(os.listdir(input_path))
  if limit==0:
    n = len(input_list)
  else:
    n = limit
  w = 160
  h = 120
  dw = 5
  dh = 4

  x_div = int(w/dw)
  y_div = 1*int(h/dh)

  for i in range(0,n):
    input_image_path = input_path + "/" + input_list[i]
    input_image = Image.open(input_image_path).copy()
    copied = Image.open(input_image_path).copy()
    copied = copied.crop((copy_x*x_div, copy_y*y_div, (copy_x+1)*x_div, (copy_y+1)*y_div))    

    for x in range(0,dw):
      xstart = x*x_div
      for y in range(0,dh):
        ystart = y*y_div
        input_image.paste(copied, (xstart, ystart))
     
    num = str(i).zfill(10)
    name = output_dir + "/" + num + ".png"
    input_image.save(name, "PNG")
    print("saved image ", name)

parser = argparse.ArgumentParser(description='image_copy')
parser.add_argument('--input', '-i', default='', help='Path to input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--limit', '-l', type=int, default=0, help='max number of images')
parser.add_argument('--cell_x', '-cx', type=int, default=1, help='which x to copy')
parser.add_argument('--cell_y', '-cy', type=int, default=1, help='which y to copy')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

copy(args.input, output_dir, args.cell_x, args.cell_y, args.limit)
