import argparse
import numpy as np
import os
from PIL import Image
# from statistics import mean 


def strong_index(rgb):
	m = np.mean(rgb)
	v = np.var(rgb)

	# print(rgb)
	# print(m)
	# print(v)
	for i in xrange(0,3):
		if rgb[i] > m and (rgb[i]-m)*(rgb[i]-m)>v:
			return(i)

	return -1


# return an image with only overpredicted colors
def color_diff(image_path):
	# create image with only the increases in blue or red
	# new_image = numpy.zeros(current_image.shape)
	# white background
	current_image = np.array(Image.open(image_path).convert('RGB'))
	new_image = np.zeros(current_image.shape)

	for i in xrange(0,current_image.shape[0]):
		for j in xrange(0,current_image.shape[1]):

			strong_color = strong_index(current_image[i, j])
			if strong_color != -1:
				new_image[i,j,strong_color] = current_image[i, j, strong_color]

	return(new_image)

def match(colors, col_image):
	new_image = np.zeros(colors.shape)

	for i in xrange(0,colors.shape[0]):
		for j in xrange(0,colors.shape[1]):
			# print(col_image[i, j])
			# print(colors[i, j])
			strong_color = strong_index(col_image[i, j])
			if strong_color != -1 and colors[i,j,strong_color] != 0:
				new_image[i,j,strong_color] = col_image[i,j,strong_color]

	return(new_image)


def overlay(matching_colors, gs_image):
	for i in xrange(0,matching_colors.shape[0]):
		for j in xrange(0,matching_colors.shape[1]):
			for color in xrange(0,3):
				if matching_colors[i,j,color] > 0:
					gs_image[i,j,color] = 255#matching_colors[i,j,color]

	return(gs_image)


# col_image_dir = directory with original color images used as input
# gs_image_dir = directory with original greyscale images used as input
def process_images(image_dir, col_image_dir, gs_image_dir, image_count, output_dir):
	# loop on images
	image_list = sorted(os.listdir(image_dir))
	gs_image_list = sorted(os.listdir(gs_image_dir))
	col_image_list = sorted(os.listdir(col_image_dir))

	if(image_count == -1):
		image_count =  len(image_list)

	diff_dir = output_dir + "/col_diff/"
	if not os.path.exists(diff_dir):
		os.makedirs(diff_dir)
	match_dir = output_dir + "/col_match/"
	if not os.path.exists(match_dir):
		os.makedirs(match_dir)
	overlay_dir = output_dir + "/col_overlay/"
	if not os.path.exists(overlay_dir):
		os.makedirs(overlay_dir)

	im_index = 0
	for image_file in image_list[:image_count]:
		image_path = os.path.join(image_dir, image_file)
		print("read ", image_path)

		# find colored pixels in black and white output
		colors = color_diff(image_path)
		# save it
		image_array = Image.fromarray(colors.astype('uint8'), 'RGB')
		name = diff_dir + image_file
		image_array.save(name)
		print("saved image ", name)

		# check if colors match with original color input
		col_image_path = os.path.join(col_image_dir, col_image_list[im_index])
		# print("col_image_path ", col_image_path)
		col_image = np.array(Image.open(col_image_path).convert('RGB'))
		matching_colors = match(colors, col_image) 
		# save it
		image_array = Image.fromarray(matching_colors.astype('uint8'), 'RGB')
		name = match_dir + image_file
		image_array.save(name)
		print("saved image ", name)

		# paste color on black and white input
		gs_image_path = os.path.join(gs_image_dir, gs_image_list[im_index])
		gs_image = np.array(Image.open(gs_image_path).convert('RGB'))
		overlay_image = overlay(matching_colors, gs_image) 
		# save it
		image_array = Image.fromarray(overlay_image.astype('uint8'), 'RGB')
		name = overlay_dir + image_file
		image_array.save(name)
		print("saved image ", name)

		im_index = im_index + 1


parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('image_dir', action='store', nargs='?', help='Path to prednet output images')
parser.add_argument('col_image_dir', action='store', nargs='?', help='Path to original color images used as input')
parser.add_argument('gs_image_dir', action='store', nargs='?', help='Path to original greyscale images used as input')
parser.add_argument('output_dir', action='store', nargs='?', help='output directory')
parser.add_argument('n_images', action='store', nargs='?', default=-1,
                    type=int, help='optional: total number of images to use.')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

process_images(args.image_dir, args.col_image_dir, args.gs_image_dir, args.n_images, args.output_dir)