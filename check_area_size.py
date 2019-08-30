# check the size of the area before it is predicted red or blue
# also check the intensity of the greyscale

import argparse
import csv
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
	v = 30#np.var(rgb)

	for i in range(0,3):
		if rgb[i] > m and (rgb[i]-m)*(rgb[i]-m)>v:
			return(i)

	return -1

def area_size(image_path, i, j):
	# segment the image
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

	gs = image[i,j]
	error = 25 # out of 256
	temp = gs-error
	if temp < 0:
		temp = 0
	gs_min = np.array([temp])

	temp = gs+error
	if temp > 255:
		temp = 255
	gs_max = np.array([temp])

	mask = cv2.inRange(image, gs_min, gs_max)
	result = cv2.bitwise_and(image, image, mask=mask)

	current_output, num_ids = label(result)
	current_id = current_output[i,j].item()

	selected_area = (current_output == current_id)
	area = np.count_nonzero(selected_area)

	return(area, gs)

def remove_area(image, i, j, color_index):
	colors = image[i,j]
	error = 20 # out of 256
	temp = colors[color_index]-error
	if temp < 0:
		temp = 0
	gs_min = np.zeros((3))
	gs_min[color_index] = temp

	temp = colors[color_index]+error
	if temp > 255:
		temp = 255
	gs_max = np.ones((3))*255
	gs_max[color_index] = temp

	mask = cv2.inRange(image, gs_min, gs_max)
	# plt.imshow(mask)
	# plt.show()

	current_output, num_ids = label(mask)
	# plt.imshow(current_output)
	# plt.show()
	# print(current_output[i,j])
	current_id = current_output[i,j].item()
	mask = cv2.inRange(current_output, current_id, current_id)
	# plt.imshow(mask)
	# plt.show()
	unselected_area = cv2.bitwise_not(mask)
	# plt.imshow(unselected_area)
	# plt.show()

	new_image = cv2.bitwise_and(image, image, mask=unselected_area)

	# plt.imshow(new_image)
	# plt.show()

	return(new_image)

# previous_gs_images = paths [t0,t1,t2]
def area_change(predicted_image, previous_gs_images, real_image, writer):
	#print(previous_gs_images)

	modifed_prediction = predicted_image

	for i in range(0,predicted_image.shape[0]):
			for j in range(0,predicted_image.shape[1]):

				# pass everything that is not black
				if(real_image[i,j,0] > 50 or real_image[i,j,1] > 50 or real_image[i,j,2] > 50):
					pass

				strong_color = strong_index(modifed_prediction[i, j])
				if strong_color != -1:
					#claculate previous area size
					area0, gs0 = area_size(previous_gs_images[0], i, j)
					area1, gs1 = area_size(previous_gs_images[1], i, j)
					area2, gs2 = area_size(previous_gs_images[2], i, j)

					# rows
					result = [strong_color, modifed_prediction[i, j][strong_color], gs0, gs1, gs2, area0, area1, area2]
					writer.writerow(result)

					# remove this area from the image
					modifed_prediction = remove_area(modifed_prediction, i, j, strong_color)


# predictions_dir: predicted images
# gs_image_dir: greyscale input
def process_images(predictions_dir, gs_image_dir, real_image_dir, output_dir, image_count):

	predictions_list = sorted(os.listdir(predictions_dir))
	gs_image_list = sorted(os.listdir(gs_image_dir))
	real_image_list = sorted(os.listdir(real_image_dir))

	if(image_count == -1):
		image_count =  len(predictions_list)

	save_file = output_dir + "/area_analysis.csv"
	fieldnames = ['color_index','color_value','value_t0','value_t1','value_t2','area_0','area_1','area_2']

	with open(save_file, mode='w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(fieldnames)

		im_index = 0

		# use prediction of t+2 (==prediction at t+1)
		# and real images at t and t+1
		for image_file in predictions_list[2:2+image_count]:
			prediction_path = os.path.join(predictions_dir, image_file)
			print("read ", prediction_path)
			predicted_image = np.array(Image.open(prediction_path).convert('RGB'))

			previous_gs_images = []
			gs_path = os.path.join(gs_image_dir, gs_image_list[im_index])
			previous_gs_images.append(gs_path)
			gs_path = os.path.join(gs_image_dir, gs_image_list[im_index+1])
			previous_gs_images.append(gs_path)
			gs_path = os.path.join(gs_image_dir, gs_image_list[im_index+2])
			previous_gs_images.append(gs_path)


			real_image = np.array(Image.open(os.path.join(real_image_dir, real_image_list[im_index+2])).convert('RGB'))
			# find area sizes changes
			areas = area_change(predicted_image, previous_gs_images, real_image, writer)
			
			im_index = im_index + 1



parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('predictions_dir', action='store', nargs='?', help='Path to prednet output images')
parser.add_argument('gs_image_dir', action='store', nargs='?', help='Path to original greyscale images used as input')
parser.add_argument('real_image_dir', action='store', nargs='?', help='Path to original images')
parser.add_argument('output_dir', action='store', nargs='?', help='output directory')
parser.add_argument('--n_images', '--n', default=-1, type=int, help='number of images to process')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

process_images(args.predictions_dir, args.gs_image_dir, args.real_image_dir, args.output_dir, args.n_images)
