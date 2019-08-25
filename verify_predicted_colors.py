import argparse
import numpy as np
import os
from PIL import Image

# return the strongest color
def strong_index(rgb):
	m = np.mean(rgb)
	v = np.var(rgb)

	for i in xrange(0,3):
		if rgb[i] > m and (rgb[i]-m)*(rgb[i]-m)>v:
			return(i)

	return -1


# return an image with only increased colors
# image_paths = [input, output]
def color_compensated(image_paths):
	print(image_paths)
	# create image with only the increases in r g or b
	# new_image = numpy.zeros(current_image.shape)
	image0 = np.array(Image.open(image_paths[0]).convert('RGB'))
	image1 = np.array(Image.open(image_paths[1]).convert('RGB'))
	new_image = np.zeros(image0.shape)

	for i in xrange(0,image0.shape[0]):
		for j in xrange(0,image0.shape[1]):
			color_increase = image1[i,j] - image0[i,j]

			for c in xrange(0,3):
				if(color_increase[c] > 0 ):
					new_image[i,j,c] = image1[i, j, c]

	return(new_image)


# return an image with only overpredicted colors
def color_diff(image_path):
	# create image with only the strongest predicted in r,g and b
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

# col_images = [t-1, t]
# TODO assess which channel gives the most reliable values (statistically)
def match_increase(colors, col_images):
	new_image = np.zeros(colors.shape)

	for i in xrange(0,colors.shape[0]):
		for j in xrange(0,colors.shape[1]):

			# new_image[i,j] = col_images[1][i,j] - col_images[0][i,j]
			color_increase = col_images[1][i,j] - col_images[0][i,j]
			for c in xrange(0,3):
				if (colors[i,j,c] >= 0 and color_increase[c] > 0):
					new_image[i,j,c] = colors[i,j,c] #col_images[1][i,j,c]

	return(new_image)


def overlay(matching_colors, gs_image):
	for i in xrange(0,matching_colors.shape[0]):
		for j in xrange(0,matching_colors.shape[1]):
			for color in xrange(0,3):
				if matching_colors[i,j,color] > 0:
					gs_image[i,j,color] = matching_colors[i,j,color]

	return(gs_image)



# image_dir = learned images
# col_image_dir = directory with original color images used as input
# gs_image_dir = directory with original greyscale images used as input
def process_increases(image_dir, col_image_dir, gs_image_dir, image_count, output_dir):
	# loop on images
	image_list = sorted(os.listdir(image_dir))
	gs_image_list = sorted(os.listdir(gs_image_dir))
	col_image_list = sorted(os.listdir(col_image_dir))

	if(image_count == -1):
		image_count =  len(image_list)

	diff_dir = output_dir + "/col_diff_strong/"
	if not os.path.exists(diff_dir):
		os.makedirs(diff_dir)
	match_dir = output_dir + "/col_match_increase/"
	if not os.path.exists(match_dir):
		os.makedirs(match_dir)
	overlay_dir = output_dir + "/col_overlay_increase/"
	if not os.path.exists(overlay_dir):
		os.makedirs(overlay_dir)

	im_index = 0
	col_images = []
	# input, output
	learned_image_paths = ["",""]

	for image_file in image_list[:image_count]:
		learned_image_path = os.path.join(image_dir, image_file)
		print("read ", learned_image_path)

		col_image_path = os.path.join(col_image_dir, col_image_list[im_index])
		col_image = np.array(Image.open(col_image_path).convert('RGB'))
		
		gs_image_path = os.path.join(gs_image_dir, gs_image_list[im_index])

		learned_image_paths[0] = gs_image_path
		learned_image_paths[1] = learned_image_path
		if(im_index>0):
			# find colored pixels in black and white output
			colors = color_diff(learned_image_path)
			# save it
			image_array = Image.fromarray(colors.astype('uint8'), 'RGB')
			name = diff_dir + image_file
			image_array.save(name)
			print("saved image ", name)

			# check if colors match with color increase in original input
			col_images[1] = col_image
			matching_colors = match_increase(colors, col_images) 
			# save it
			image_array = Image.fromarray(matching_colors.astype('uint8'), 'RGB')
			name = match_dir + image_file
			image_array.save(name)
			print("saved image ", name)

			# paste color on black and white input
			gs_image = np.array(Image.open(gs_image_path).convert('RGB'))
			print("grey")
			print(gs_image_path)
			overlay_image = overlay(matching_colors, gs_image) 
			# save it
			image_array = Image.fromarray(overlay_image.astype('uint8'), 'RGB')
			name = overlay_dir + image_file
			image_array.save(name)
			print("saved image ", name)
		else:
			# preallocate size
			col_images.append(col_image)
			col_images.append(col_image)

		im_index = im_index + 1
		col_images[0] = col_image
		learned_image_paths[0] = learned_image_path

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
	match_dir = output_dir + "/col_match_increase/"
	if not os.path.exists(match_dir):
		os.makedirs(match_dir)
	overlay_dir = output_dir + "/col_match_overlay/"
	if not os.path.exists(overlay_dir):
		os.makedirs(overlay_dir)

	im_index = 0
	col_images = []
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

		# # check if colors match with original color input
		# col_image_path = os.path.join(col_image_dir, col_image_list[im_index])
		# # print("col_image_path ", col_image_path)
		# col_image = np.array(Image.open(col_image_path).convert('RGB'))
		# matching_colors = match(colors, col_image) 
		# # save it
		# image_array = Image.fromarray(matching_colors.astype('uint8'), 'RGB')
		# name = match_dir + image_file
		# image_array.save(name)
		# print("saved image ", name)


		col_image_path = os.path.join(col_image_dir, col_image_list[im_index])
		col_image = np.array(Image.open(col_image_path).convert('RGB'))
		if(im_index>0):
			# check if colors match with color increase in original input
			col_images[1] = col_image
			matching_colors = match_increase(colors, col_images) 
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
		else:
			# preallocate size
			col_images.append(col_image)
			col_images.append(col_image)

		im_index = im_index + 1
		col_images[0] = col_image


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

#process_images(args.image_dir, args.col_image_dir, args.gs_image_dir, args.n_images, args.output_dir)
process_increases(args.image_dir, args.col_image_dir, args.gs_image_dir, args.n_images, args.output_dir)