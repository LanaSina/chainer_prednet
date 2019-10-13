# check the size of the area before it is predicted red or blue
# also check the intensity of the greyscale

import argparse
from array import array
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

# black and white cv2 image
def area_size(image, i, j):
	
	# segment the image
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
	# plt.imshow(mask)
	# plt.show()

	result = cv2.bitwise_and(image, image, mask=mask)

	current_output, num_ids = label(result)
	current_id = current_output[i,j].item()
	# print(current_id)
	# print(current_id)
	# plt.imshow(current_output)
	# plt.show()


	selected_area = (current_output == current_id)
	area = np.count_nonzero(selected_area)

	return(area, gs)


# calculate size, center, and average rgb
# area: 1--256
def get_area_parameters(image):

	size = 0.0
	center = np.array([0,0])
	av_rgb = np.array([0,0,0])

	for i in range(0,image.shape[0]):
		for j in range(0,image.shape[1]):
			pixel = image[i,j] 

			# ignore areas that have been removed
			if( (1.0*pixel[0]+pixel[1]+pixel[2]) == 0):
				continue

			size = size +1
			av_rgb = av_rgb + pixel
			center = center + [i,j]

	center = [center[0]/size, center[1]/size]
	av_rgb = av_rgb/size

	return(size, center, av_rgb)


# image = something between 1 and 256
def select_area(image, i, j, show):

	# segment the image
	rgb = 1.0*image[i,j]
	error = 40
	temp = rgb - error

	for k in range(0,3):
		# avoid removed areas
		if temp[k] < 1:
			temp[k] = 1.0
	gs_min = temp

	rgb = 1.0*image[i,j].copy()
	temp = rgb + error
	for k in range(0,3):
		if temp[k] > 255:
			temp[k] = 255
	gs_max = np.array(temp)

	mask = cv2.inRange(image, gs_min, gs_max)

	result = cv2.bitwise_and(image, image, mask=mask)

	current_output, num_ids = label(result)
	current_id = current_output[i,j]

	selected_mask = cv2.inRange(current_output, current_id, current_id)
	new_image = cv2.bitwise_and(image, image, mask=selected_mask)
	if (show):
		w=10
		h=10
		fig=plt.figure(figsize=(8, 8))
		columns = 1
		rows = 3
		fig.add_subplot(rows, columns, 1)
		plt.imshow(image)
		fig.add_subplot(rows, columns, 2)
		plt.imshow(selected_mask)
		fig.add_subplot(rows, columns, 3)
		plt.imshow(new_image)

		plt.show()


	return(new_image, selected_mask)


def select_area_from_mask(image, mask, show):

	new_image = cv2.bitwise_and(image, image, mask=mask)
	if (show):
		plt.imshow(new_image)
		plt.show()

	return(new_image)


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

# previous_images, output_dir, writer
def areas_changes(im_index, previous_images, output_dir, writer):

	# find an area
	# calculate its center, size, and average rgb
	# find same area in next image using center value
	# calculate its center, size, and average rgb
	# find same area in next next image
	# calculate its size and average rgb
	# record all values
	# remove area from original image at t0, look for next area
	# repeat until whole image at t0 is processed

	# print("previous_images", previous_images)

	image_t0 = cv2.imread(previous_images[0])
	image_t0 = cv2.cvtColor(image_t0, cv2.COLOR_BGR2RGB)
	image_t1 = cv2.imread(previous_images[1])
	image_t1 = cv2.cvtColor(image_t1, cv2.COLOR_BGR2RGB)
	image_t2 = cv2.imread(previous_images[2])
	image_t2 = cv2.cvtColor(image_t2, cv2.COLOR_BGR2RGB)

	show = False
	area_id = 0
	for i in range(0,image_t0.shape[0]):
			for j in range(0,image_t0.shape[1]):

				pixel0 = image_t0[i,j] 
				pixel1 = image_t1[i,j] 
				pixel2 = image_t2[i,j] 

				# ignore areas that have been removed/ and by default, ignore black areas too
				if( pixel0[0]<=1 or pixel0[1]<=1 or pixel0[2] <= 1):
					continue
				if( pixel1[0]<=1 or pixel1[1]<=1 or pixel1[2] <= 1):
					continue
				if( pixel2[0]<=1 or pixel2[1]<=1 or pixel2[2] <= 1):
					continue

				# get an array only containing this area
				selected_area_0, _ = select_area(image_t0, i, j, show)
				size_t0, center_t0, rgb_t0 = get_area_parameters(selected_area_0) 
				# get corresponding area in next image
				selected_area_1, _ = select_area(image_t1, int(center_t0[0]), int(center_t0[1]), show)
				size_t1, center_t1, rgb_t1 = get_area_parameters(selected_area_1) 
				# get corresponding area in next image
				selected_area_2, _ = select_area(image_t2, int(center_t1[0]), int(center_t1[1]), show)
				size_t2, center_t2, rgb_t2 = get_area_parameters(selected_area_2) 

				# if (size_t0>size_t1 and size_t1>size_t2):# and rgb_t2[0]<rgb_t0[0]):
				# 	w=10
				# 	h=10
				# 	fig=plt.figure(figsize=(8, 8))
				# 	columns = 3
				# 	rows = 1
				# 	fig.add_subplot(rows, columns, 1)
				# 	plt.imshow(selected_area_0)
				# 	fig.add_subplot(rows, columns, 2)
				# 	plt.imshow(selected_area_1)
				# 	fig.add_subplot(rows, columns, 3)
				# 	plt.imshow(selected_area_2)

				# 	plt.show()


				# rows
				# fieldnames = ['image_t0', 'area_id','size_t0','size_t1','size_t2','rgb_t0','rgb_t1','rgb_t2']
				result = [im_index, area_id, size_t0, size_t1, size_t2]
				result.extend(rgb_t0)
				result.extend(rgb_t1)
				result.extend(rgb_t2)
				writer.writerow(result)

				# remove this area from the image
				mask = cv2.inRange(selected_area_0, 0, 0)
				image_t0 = cv2.bitwise_and(image_t0, image_t0, mask=mask)
				area_id = area_id + 1

# add +1 to rgb values that are == 0
def add_one_to_0s(image):
	for i in range(0,image.shape[0]):
			for j in range(0,image.shape[1]):
				pixel = image[i,j]
				for p in range(0,3):
					if(pixel[p]==0):
						pixel[p] = 1
	return(image)
				
# previous_images, output_dir, writer
def fixed_areas_changes(im_index, previous_images, output_dir, writer):

	image_t0 = cv2.imread(previous_images[0])
	image_t0 = cv2.cvtColor(image_t0, cv2.COLOR_BGR2RGB)
	image_t1 = cv2.imread(previous_images[1])
	image_t1 = cv2.cvtColor(image_t1, cv2.COLOR_BGR2RGB)
	image_t2 = cv2.imread(previous_images[2])
	image_t2 = cv2.cvtColor(image_t2, cv2.COLOR_BGR2RGB)

	image_t0 = add_one_to_0s(image_t0)
	image_t1 = add_one_to_0s(image_t1)
	image_t2 = add_one_to_0s(image_t2)

	# plt.imshow(image_t2)
	# plt.show()

	show = False
	area_id = 0
	for i in range(0,image_t0.shape[0]):
			for j in range(0,image_t0.shape[1]):

				pixel0 = image_t0[i,j] 
				pixel1 = image_t1[i,j] 
				pixel2 = image_t2[i,j] 

				# ignore areas that have been removed/ and by default, ignore black areas too
				if( pixel0[0]<=1 or pixel0[1]<=1 or pixel0[2] <= 1):
					continue
				if( pixel1[0]<=1 or pixel1[1]<=1 or pixel1[2] <= 1):
					continue
				if( pixel2[0]<=1 or pixel2[1]<=1 or pixel2[2] <= 1):
					continue

				# get an array only containing this area, from the 3rd image
				selected_area_2, mask = select_area(image_t2, i, j, show)
				size_t2, center_t2, rgb_t2 = get_area_parameters(selected_area_2) 
				# get this exact same area from image 1 and 0
				selected_area_1 = select_area_from_mask(image_t1, mask, show)
				size_t1, center_t1, rgb_t1 = get_area_parameters(selected_area_1) 
				selected_area_0 = select_area_from_mask(image_t0, mask, show)
				size_t0, center_t0, rgb_t0 = get_area_parameters(selected_area_0)  

				print("s0 ", size_t0, " s1 ", size_t1, " s2 ", size_t2)

				if (size_t0>size_t1):# and size_t1<size_t2):# and rgb_t2[0]<rgb_t0[0]):
					w=10
					h=10
					fig=plt.figure(figsize=(8, 8))
					columns = 1
					rows = 3
					fig.add_subplot(rows, columns, 1)
					plt.imshow(selected_area_0)
					fig.add_subplot(rows, columns, 2)
					plt.imshow(selected_area_1)
					fig.add_subplot(rows, columns, 3)
					plt.imshow(selected_area_2)

				plt.show()

				# rows
				# fieldnames = ['image_t0', 'area_id','size_t0','size_t1','size_t2','rgb_t0','rgb_t1','rgb_t2']
				result = [im_index, area_id, size_t0, size_t1, size_t2]
				result.extend(rgb_t0)
				result.extend(rgb_t1)
				result.extend(rgb_t2)
				writer.writerow(result)

				# remove this area from the image
				mask = cv2.inRange(selected_area_0, 0, 0)
				image_t0 = cv2.bitwise_and(image_t0, image_t0, mask=mask)
				area_id = area_id + 1


# previous_gs_images = paths [t0,t1,t2]
def area_change(predicted_image, previous_gs_images, real_image, writer):
	#print(previous_gs_images)

	modifed_prediction = predicted_image

	read_gs_images = []
	for path in previous_gs_images:
		read_gs_images.extend([cv2.imread(path, cv2.IMREAD_GRAYSCALE)])

	for i in range(0,predicted_image.shape[0]):
			for j in range(0,predicted_image.shape[1]):

				# pass everything that is not black
				if(real_image[i,j,0] > 50 or real_image[i,j,1] > 50 or real_image[i,j,2] > 50):
					pass

				strong_color = strong_index(modifed_prediction[i, j])
				if strong_color != -1:
					#claculate previous area size
					area0, gs0 = area_size(read_gs_images[0], i, j)
					area1, gs1 = area_size(read_gs_images[1], i, j)
					area2, gs2 = area_size(read_gs_images[2], i, j)

					# rows
					result = [strong_color, modifed_prediction[i, j][strong_color], gs0, gs1, gs2, area0, area1, area2]
					writer.writerow(result)

					# remove this area from the image
					modifed_prediction = remove_area(modifed_prediction, i, j, strong_color)


def has_flickered(read_gs_images, i, j):
	col0 = read_gs_images[0][i,j]
	col1 = read_gs_images[1][i,j]
	col2 = read_gs_images[2][i,j]

	# very black = 5
	# kinda black = 120
	# very white = 255
	black = 128 #120
	white = 128 #250

	if(col0<black and col1>white and col2<black):
		return True

	return False


def flickers(previous_gs_images, predicted_image, output_dir, writer):
	read_gs_images = []

	for path in previous_gs_images:
		read_gs_images.extend([cv2.imread(path, cv2.IMREAD_GRAYSCALE)])

	for i in range(0,predicted_image.shape[0]):
			for j in range(0,predicted_image.shape[1]):

				# check for flicker
				if(has_flickered(read_gs_images, i, j)):
					# calculate area change
					area0, gs0 = area_size(read_gs_images[0], i, j)
					area1, gs1 = area_size(read_gs_images[1], i, j)
					area2, gs2 = area_size(read_gs_images[2], i, j)

					# rows
					result = [previous_gs_images[2], gs0, gs1, gs2, area0, area1, area2]
					result.extend(predicted_image[i,j])
					writer.writerow(result)

# prediction_dir: predicted images
# gs_image_dir: greyscale input
def process_flicker(prediction_dir, gs_image_dir, real_image_dir, output_dir, image_count):
	predictions_list = sorted(os.listdir(prediction_dir))
	gs_image_list = sorted(os.listdir(gs_image_dir))
	real_image_list = sorted(os.listdir(real_image_dir))

	if(image_count == -1):
		image_count =  len(predictions_list)

	save_file = output_dir + "/flicker_analysis_original.csv"
	print(save_file)
	fieldnames = ['last_gs_image','value_t0','value_t1','value_t2','area_0','area_1','area_2', 'predicted_r', 'predicted_g', 'predicted_b']

	with open(save_file, mode='w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(fieldnames)

		im_index = 0

		# use prediction of t+2 (==prediction at t+1)
		# and real images at t and t+1
		for image_file in predictions_list[2:2+image_count]:
			prediction_path = os.path.join(prediction_dir, image_file)
			print("read ", prediction_path)
			predicted_image = np.array(Image.open(prediction_path).convert('RGB'))

			previous_gs_images = []
			gs_path = os.path.join(gs_image_dir, gs_image_list[im_index])
			previous_gs_images.append(gs_path)
			gs_path = os.path.join(gs_image_dir, gs_image_list[im_index+1])
			previous_gs_images.append(gs_path)
			gs_path = os.path.join(gs_image_dir, gs_image_list[im_index+2])
			previous_gs_images.append(gs_path)

			# print("read ", previous_gs_images)



			#real_image = np.array(Image.open(os.path.join(real_image_dir, real_image_list[im_index+2])).convert('RGB'))
			#find flickers
			flickers(previous_gs_images, predicted_image, output_dir, writer)
			
			im_index = im_index + 1

# prediction_dir: predicted images
# gs_image_dir: greyscale input
def process_images(prediction_dir, gs_image_dir, real_image_dir, output_dir, image_count):

	predictions_list = sorted(os.listdir(prediction_dir))
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
			prediction_path = os.path.join(prediction_dir, image_file)
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


def record_area_changes(real_image_dir, prediction_dir, output_dir, image_count):
	real_image_list = sorted(os.listdir(real_image_dir))
	#prediction_list = sorted(os.listdir(prediction_dir))


	if(image_count == -1):
		image_count =  len(real_image_list)

	save_file = output_dir + "/_temp_fixed_area_change_analysis.csv"
	print(save_file)
	fieldnames = ['image_0','area_id','size_t0','size_t1','size_t2','r0','g0','b0','r1','g1','b1','r2','g2','b2']

	with open(save_file, mode='w') as csv_file:

		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(fieldnames)
		im_index = 0
		print("all ", real_image_list[2:image_count])
		for image_file in real_image_list[2:image_count]:
			path = os.path.join(real_image_dir, image_file)
			print("read ", path)
			# predicted_image = np.array(Image.open(prediction_path).convert('RGB'))

			previous_images = []
			# previous_path = os.path.join(real_image_dir, real_image_list[im_index])
			# previous_images.append(previous_path)
			# previous_path = os.path.join(real_image_dir, real_image_list[im_index+1])
			# previous_images.append(previous_path)
			# previous_images.append(path)
			previous_path = os.path.join(real_image_dir, real_image_list[im_index])
			previous_images.append(previous_path)
			previous_path = os.path.join(real_image_dir, real_image_list[im_index+1])
			previous_images.append(previous_path)
			# last image is the prediction
			#prediction_dir
			previous_path = os.path.join(real_image_dir, real_image_list[im_index+2]) #prediction_list[im_index+1])
			previous_images.append(previous_path)

			print("previous images ------ ")
			print(previous_images)

			fixed_areas_changes(im_index,previous_images, output_dir, writer)
			
			im_index = im_index + 1
			print("here")


## record pixel mean color at t-2 and t-1
def record_relative_colors(real_image_dir, prediction_dir, output_dir, image_count):
	real_image_list = sorted(os.listdir(real_image_dir))
	#prediction_list = sorted(os.listdir(prediction_dir))


	if(image_count == -1):
		image_count =  len(real_image_list)

	save_file = output_dir + "/_temp_fixed_area_change_analysis.csv"
	print(save_file)
	fieldnames = ['image_0','area_id','size_t0','size_t1','size_t2','r0','g0','b0','r1','g1','b1','r2','g2','b2']

	with open(save_file, mode='w') as csv_file:

		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(fieldnames)
		im_index = 0
		print("all ", real_image_list[2:image_count])
		for image_file in real_image_list[2:image_count]:
			path = os.path.join(real_image_dir, image_file)
			print("read ", path)
			# predicted_image = np.array(Image.open(prediction_path).convert('RGB'))

			previous_images = []
			# previous_path = os.path.join(real_image_dir, real_image_list[im_index])
			# previous_images.append(previous_path)
			# previous_path = os.path.join(real_image_dir, real_image_list[im_index+1])
			# previous_images.append(previous_path)
			# previous_images.append(path)
			previous_path = os.path.join(real_image_dir, real_image_list[im_index])
			previous_images.append(previous_path)
			previous_path = os.path.join(real_image_dir, real_image_list[im_index+1])
			previous_images.append(previous_path)
			# last image is the prediction
			#prediction_dir
			previous_path = os.path.join(real_image_dir, real_image_list[im_index+2]) #prediction_list[im_index+1])
			previous_images.append(previous_path)

			print("previous images ------ ")
			print(previous_images)

			fixed_areas_changes(im_index,previous_images, output_dir, writer)
			
			im_index = im_index + 1
			print("here")


parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('prediction_dir', action='store', nargs='?', help='Path to prednet output images')
parser.add_argument('gs_image_dir', action='store', nargs='?', help='Path to original greyscale images used as input')
parser.add_argument('real_image_dir', action='store', nargs='?', help='Path to original images')
parser.add_argument('output_dir', action='store', nargs='?', help='output directory')
parser.add_argument('--n_images', '--n', default=-1, type=int, help='number of images to process')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#process_images(args.prediction_dir, args.gs_image_dir, args.real_image_dir, args.output_dir, args.n_images)
#process_flicker(args.prediction_dir, args.gs_image_dir, args.real_image_dir, args.output_dir, args.n_images)
#process_flicker(args.real_image_dir, args.gs_image_dir, args.real_image_dir, args.output_dir, args.n_images)
record_area_changes(args.real_image_dir, args.prediction_dir, args.output_dir,  args.n_images)
record_relative_colors()

