import math
import numpy as np
import os
from PIL import Image
from random import random, shuffle


# Create an Image object from an Image
input_image  = Image.open("square-snakes.png")
output_path = "scrambled.png"
#px
cell_size = 20

#make image divisible
x_cells = math.ceil(input_image.size[0]/cell_size)
y_cells = math.ceil(input_image.size[1]/cell_size)


new_image = Image.new('RGB', (x_cells*cell_size, y_cells*cell_size))
new_image.paste(input_image)
# this inverts x and y for some reason
new_image = np.array(new_image)
scrambled = Image.new('RGB', (x_cells*cell_size, y_cells*cell_size))
scrambled = np.array(scrambled) #np.reshape(np.array(scrambled), (x_cells*cell_size,y_cells*cell_size,3))

# np inverts PIL's x and y for some reason
for y in range(x_cells):
	for x in range(y_cells):
		# scramble
		x_range = np.linspace(x*cell_size, (x+1)*cell_size, num = cell_size, endpoint=False).astype(int)
		y_range = np.linspace(y*cell_size, (y+1)*cell_size, num = cell_size, endpoint=False).astype(int)
		shuffle(x_range)
		shuffle(y_range)
		for xx in range(cell_size):
			for yy in range(cell_size):
				scrambled[x_range[xx],y_range[yy],:] = new_image[x*cell_size+xx, y*cell_size+yy,:]
	
Image.fromarray(scrambled).save(output_path, "PNG")


# if __name__ == "__main__":
# parser = argparse.ArgumentParser(
# description='PredNet')
# parser.add_argument('--images_path', '-i', default='', help='Path to input images')
# parser.add_argument('--output_dir', '-out', default= "result", help='where to save predictions')
# parser.add_argument('--sequences', '-seq', default='', help='In text mode, Path to file with list of text files, that themselves contain lists of images')
# parser.add_argument('--gpu', '-g', default=-1, type=int,
#                     help='GPU ID (negative value indicates CPU)')
# parser.add_argument('--initmodel', default='',
#                     help='Initialize the model from given file')
# parser.add_argument('--resume', default='',
#                     help='Resume the optimization from snapshot')
