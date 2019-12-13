import argparse
import numpy as np
import os
from PIL import Image


def copy_left(input_path, output_dir):

  input_list = sorted(os.listdir(input_path))
  n = len(input_list)
  w = 160

  for i in range(0,n):
    input_image_path = input_path + "/" + input_list[i]
    input_image = np.array(Image.open(input_image_path).convert('RGB'))

    final = np.zeros(input_image.shape)
    final[:,0:int(w/2),:] = input_image[:,0:int(w/2),:]
    final[:,int(w/2):w,:] = input_image[:,0:int(w/2),:]
    image_array = Image.fromarray(final.astype('uint8'), 'RGB')
    name = output_dir + "/" + input_list[i]
    image_array.save(name)
    print("saved image ", name)

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--input', '-i', default='', help='Path to input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')


args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

copy_left(args.input, output_dir)
