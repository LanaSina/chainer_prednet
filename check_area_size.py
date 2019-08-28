# check the size of the area before it is predicted red or blue
# also check the intensity of the greyscale

import argparse
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import label
#import scipy

# return the strongest color
def strong_index(rgb):
	m = np.mean(rgb)
	v = np.var(rgb)

	for i in range(0,3):
		if rgb[i] > m and (rgb[i]-m)*(rgb[i]-m)>v:
			return(i)

	return -1

def area_size(image_path, i, j):
	# segment the image
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# plt.imshow(image)
	# plt.show()
	gs = image[i,j,0]
	error = 2 # out of 256
	temp = gs-2
	if temp < 0:
		temp = 0
	gs_min = np.array([temp, temp, temp])

	temp = gs+2
	if temp > 255:
		temp = 255
	gs_max = np.array([temp, temp, temp])

	print(gs_min)
	print(gs_max)

	mask = cv2.inRange(image, gs_min, gs_max)
	result = cv2.bitwise_and(image, image, mask=mask)
	print(result)

	current_output, num_ids = label(result)
	print(current_output)

	plt.imshow(result)
	plt.show()

# previous_gs_images = paths [t0,t1]
def area_change(predicted_image, previous_gs_images):

	for i in range(0,predicted_image.shape[0]):
			for j in range(0,predicted_image.shape[1]):

				strong_color = strong_index(predicted_image[i, j])
				if strong_color != -1:
					#claculate previous area size
					print("here")
					area = area_size(previous_gs_images[0], i, j)


# predictions_dir: predicted images
# gs_image_dir: greyscale input
def process_images(predictions_dir, gs_image_dir, output_dir, image_count):

	predictions_list = sorted(os.listdir(predictions_dir))
	gs_image_list = sorted(os.listdir(gs_image_dir))

	if(image_count == -1):
		image_count =  len(image_list)

	save_file = output_dir + "/area_analysis.py"

	im_index = 0

	for image_file in predictions_list[2:2+image_count]:
		prediction_path = os.path.join(predictions_dir, image_file)
		print("read ", prediction_path)
		predicted_image = np.array(Image.open(prediction_path).convert('RGB'))

		previous_gs_images = []
		gs_path = os.path.join(gs_image_dir, gs_image_list[im_index])
		# gs_image = np.array(Image.open(gs_path).convert('RGB'))
		previous_gs_images.append(gs_path)
		gs_path = os.path.join(gs_image_dir, gs_image_list[im_index+1])
		# gs_image = np.array(Image.open(gs_path).convert('RGB'))
		previous_gs_images.append(gs_path)

		# find area sizes changes
		areas = area_change(predicted_image, previous_gs_images)
		
		# save it
		# csv
		im_index = im_index + 1



parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('predictions_dir', action='store', nargs='?', help='Path to prednet output images')
parser.add_argument('gs_image_dir', action='store', nargs='?', help='Path to original greyscale images used as input')
parser.add_argument('output_dir', action='store', nargs='?', help='output directory')
parser.add_argument('--n_images', '--n', default=-1, type=int, help='number of images to process')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

process_images(args.predictions_dir, args.gs_image_dir, args.output_dir, args.n_images)
