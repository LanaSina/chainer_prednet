import argparse
import numpy as np
import os
from PIL import Image



usage = 'Usage: python {} DATA_DIR [N_IMAGES] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate text list files image datasets',
                                 usage=usage)
parser.add_argument('data_dir', action='store', nargs=None, 
                    type=str, help='path to directory containing the input_images.')
parser.add_argument('n_images', action='store', nargs='?', default=-1,
                    type=int, help='optional: total number of images to use.')
args = parser.parse_args()


output_directory = "./greyscale_input/input_images/"
if not os.path.exists(output_directory): 
    	os.makedirs(output_directory)
    	print("created ", output_directory)

image_list = sorted(os.listdir(args.data_dir))

n_images = args.n_images
if n_images==-1:
    n_images = len(image_list)

for image_file in image_list[:n_images]:
			image_path = os.path.join(args.data_dir, image_file)
			print("read ", image_path)
			img = np.array(Image.open(image_path).convert('L'))
			stacked_img = np.stack((img,)*3, axis=-1)
			image_array = Image.fromarray(stacked_img.astype('uint8'), 'RGB')
			image_array.save(output_directory + image_file)