import argparse
import numpy as np
import os
from PIL import Image

# cuts the images into w*h 5*4 parts
# copies the 1:1 space into all slots (avoiding black lines at the top of image and boundary conditions)
def copy(input_path, output_dir):

  input_list = sorted(os.listdir(input_path))
  n = len(input_list)
  w = 160
  h = 120
  dw = 5
  dh = 4

  x_div = int(w/dw)
  y_div = 1*int(h/dh)
  copy_x = 1
  copy_y = 1 

  for i in range(0,n):
    input_image_path = input_path + "/" + input_list[i]
    input_image = np.array(Image.open(input_image_path).convert('RGB'))

    final = np.zeros(input_image.shape)
    for x in range(0,dw):
      xstart = x*x_div
      xend = (x+1)*x_div
      for y in range(0,dh):
        ystart = y*y_div
        yend = (y+1)*y_div
        final[ystart:yend,xstart:xend:] = input_image[copy_y*y_div:(copy_y+1)*y_div,copy_x*x_div:(copy_x+1)*x_div,:]
    image_array = Image.fromarray(final.astype('uint8'), 'RGB')
    name = output_dir + "/" + input_list[i]
    image_array.save(name)
    print("saved image ", name)

parser = argparse.ArgumentParser(description='image_copy')
parser.add_argument('--input', '-i', default='', help='Path to input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')


args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

copy(args.input, output_dir)
