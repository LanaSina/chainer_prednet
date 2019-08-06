import argparse
import csv
import cv2
import numpy
import os
from PIL import Image
import random

# dataset = "/bike_lines_fw/"
# image_dir = "datasets/" + dataset
# output_dir = "results/"+ dataset
output_dir = "image_analysis/black_white_next/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--image_dir', '-i', default='', help='Path to images folder')
args = parser.parse_args()

# image_list = sorted(os.listdir(image_dir))
# image_count = 23 # or image_list.length
# n = 0

# red[top, bottom] blue[top, bottom]
def top_bottom_diff(current_image, middle):
	# sum red and blue on the 2 halves
	red = [0,0]
	blue = [0,0]
	for i in xrange(0, current_image.shape[0]):
		for j in xrange(0, current_image.shape[1]):
			if i<middle: # top
				red[0] = red[0] + current_image[i, j, 0]
				blue[0] = blue[0] + current_image[i, j, 2]
			else:
				red[1] = red[1] + current_image[i, j, 0]
				blue[1] = blue[1] + current_image[i, j, 2]

	return(red, blue)

def motion_half(current_image, middle):
	# sum red and blue on the 2 halves
	red = [0,0]
	blue = [0,0]
	for i in xrange(0, current_image.shape[0]):
		for j in xrange(0, current_image.shape[1]):
			if j<middle: # left
				red[0] = red[0] + current_image[i, j, 0]
				blue[0] = blue[0] + current_image[i, j, 2]
			else:
				red[1] = red[1] + current_image[i, j, 0]
				blue[1] = blue[1] + current_image[i, j, 2]


	red[0] = red[0] / (current_image.shape[0]*current_image.shape[1])
	red[1] = red[1] / (current_image.shape[0]*current_image.shape[1])
	blue[0] = blue[0] / (current_image.shape[0]*current_image.shape[1])
	blue[1] = blue[1] / (current_image.shape[0]*current_image.shape[1])

	return(red, blue)

# [red, blue]
def red_blue_average(current_image):
	# average red and blue vlaues
	result = [0,0]
	for i in xrange(0, current_image.shape[0]):
		for j in xrange(0, current_image.shape[1]):
			result[0] = result[0] + current_image[i, j, 0]
			result[1] = result[1] + current_image[i, j, 2]
	
	result[0] = result[0] / (current_image.shape[0]*current_image.shape[1])
	result[1] = result[1] / (current_image.shape[0]*current_image.shape[1])

	return(result)

#average by column
def red_blue_green(current_image, output_file):
	print(current_image.shape)
	fieldnames = ['red','blue','green']
	with open(output_dir+output_file, mode='w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(fieldnames)

		result = [0,0,0]
		for j in xrange(0, current_image.shape[1]):
			for i in xrange(0, current_image.shape[0]):
				result[0] = result[0] + current_image[i, j, 0]
				result[1] = result[1] + current_image[i, j, 1]
				result[2] = result[2] + current_image[i, j, 2]
			result[0] = result[0] / (current_image.shape[0])
			result[1] = result[1] / (current_image.shape[0])
			result[2] = result[2] / (current_image.shape[0])
			writer.writerow(result)
	print("wrote in ", output_dir)


def red_blue_diff(current_image, next_image):
	# create image with only the increases in blue or red
	# new_image = numpy.zeros(current_image.shape)
	# white background
	new_image = numpy.ones(current_image.shape)
	new_image = new_image*255
	#new_image = current_image;

	for i in xrange(0,current_image.shape[0]):
		for j in xrange(0,current_image.shape[1]):

			rdiff = 1.0*next_image[i, j, 0] - current_image[i, j, 0]
			# gdiff = 1.0*next_image[i, j, 1] - current_image[i, j, 1]
			bdiff = 1.0*next_image[i, j, 2] - current_image[i, j, 2]

			lim = 0
			# if (rdiff>bdiff):
			# if (bdiff<=lim and rdiff>lim):
			if (rdiff>0 and rdiff<100):
				new_image[i, j, 0] = rdiff
				new_image[i, j, 1] = 0
				new_image[i, j, 2] = 0
			# if (bdiff>rdiff):
			# if (rdiff<lim and bdiff>lim):
			# if (bdiff>0 and bdiff<100):
			# 	new_image[i, j, 0] = 0
			# 	new_image[i, j, 1] = 0
			# 	new_image[i, j, 2] = bdiff
			# if (gdiff>bdiff):
			# 	new_image[i, j, 1] = gdiff

	return new_image

# count the ratio of blue and red adter bblack-white alternation in a pixel
def sample_average_color(image_dir, output_dir):
	image_list = sorted(os.listdir(image_dir))
	image_count = len(image_list)

	# sample from this square
	w = 0
	h = 90/2
	size = 20

	#better to open several files... but how
	output_file = "bike_4h_left_sample.csv"
	fieldnames = ['r','g','b']

	# output_file = "fpsi_transitions.csv"
	# fieldnames = ['black_tr', 'white_tr']
	with open(output_dir+output_file, mode='w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(fieldnames)

		for image_file in image_list[:image_count]:
			image_path = os.path.join(image_dir, image_file)
			
			# h, w, color
			current_image = numpy.array(Image.open(image_path).convert('RGB'))
			r = 0.0
			g = 0.0
			b = 0.0
			for i in range(h,h+size):
				for j in range(w,w+size):
					r = r + current_image[i, j, 0]
					g = g + current_image[i, j, 1]
					b = b + current_image[i, j, 2]

			r = r/(size*size)
			g = g/(size*size)
			b = b/(size*size)
			row = [r,g,b]
			writer.writerow(row)

	print("done")

def rgb_surround(i,j, size, image):
	rgb = numpy.zeros(3)
	max_i = image.shape[0]
	max_j = image.shape[1]

	i_start = i - size;
	if i_start <0:
		i_start = 0
	j_start = j -size
	if j_start <0:
		j_start = 0

	i_end = i+size
	if i_end >max_i:
		i_end = max_i
	j_end = j+size
	if j_end >max_j:
		j_end = max_j

	for ii in xrange(i_start,i_end):
		for jj in xrange(j_start,j_end):
			if (ii!=i) and (jj!=j):
				rgb = rgb + image[ii,jj]

	rgb = rgb / ( (i_end-i_start)*(j_end-j_start) - 1)
	return rgb

# count the ratio of blue and red adter bblack-white alternation in a pixel
def black_white_next(image_dir, output_dir):
	image_list = sorted(os.listdir(image_dir))
	image_count =  10#len(image_list)

	# choose some random pixels
	w = 160
	h = 90
	pixels_count = w*h
	coordinates = numpy.zeros((pixels_count,2))
	previous_images = [numpy.zeros((h,w,3)),numpy.zeros((h,w,3)),numpy.zeros((h,w,3))]

	# for i in xrange(0,pixels_count):
	# 	# row
	# 	coordinates[i][0] = random.randrange(0,h)
	# 	# col
	# 	coordinates[i][1] = random.randrange(0,w)

	for i in range(0,h):
		for j in range(0,w):
			n = i*w + j
			# row
			coordinates[n][0] = i
			# col
			coordinates[n][1] = j

	t = 3
	black = numpy.zeros((pixels_count,t))
	white = numpy.zeros((pixels_count,t))
	col_mean = numpy.zeros((pixels_count,t))
	b_t = 200
	w_t = 55
	blue_t = 0
	red_t = 0
	variation = 0.0

	#better to open several files... but how
	output_file = "bike_15m_flicker_surround.csv"
	# fieldnames = ['bw_r','bw_b','wb_r','wb_b',"bw_surround_r","bw_surround_g","bw_surround_b","wb_surround_r","wb_surround_g","wb_surround_b"]

	fieldnames = ['bw_r','bw_b','wb_r','wb_b']
	# fieldnames = []
	# prefix = "bw"
	# for i in range(0,3):
	# 	fieldnames.append(prefix + "_r" + str(i))
	# 	fieldnames.append(prefix + "_g" + str(i))
	# 	fieldnames.append(prefix + "_b" + str(i))
	# prefix = "wb"
	# for i in range(0,3):
	# 	fieldnames.append(prefix + "_r" + str(i))
	# 	fieldnames.append(prefix + "_g" + str(i))
	# 	fieldnames.append(prefix + "_b" + str(i))

	# output_file = "fpsi_transitions.csv"
	# fieldnames = ['black_tr', 'white_tr']
	started = False
	with open(output_dir+output_file, mode='w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(fieldnames)

		for image_file in image_list[:image_count]:
			image_path = os.path.join(image_dir, image_file)
			print("read ", image_path)

			# h, w, color
			current_image = numpy.array(Image.open(image_path).convert('RGB'))

			# current_image = cv2.imread(image_path)
			# current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) 

			# w, h ...
			new_image = numpy.ones(current_image.shape)*255
			pixel_index = 0
			# (black->white , white->black)
			blue_sum = [0,0]
			red_sum = [0,0]
			green_sum = [0,0]
			bw_sum = 1 # avoid /0
			wb_sum = 1
			#bw, wb
			color_tracks = numpy.zeros((2,3,3))
			rgb_s = numpy.zeros((2,3))


			for time_index in range(0,t-1):
				previous_images[time_index] = previous_images[time_index+1]
			previous_images[2] = current_image

			for pixel in coordinates:
			
				i = int(pixel[0])
				j = int(pixel[1])

				# !!!!! avoid weird typing
				r = 1.0*current_image[i, j, 0]
				g = 1.0*current_image[i, j, 1]
				b = 1.0*current_image[i, j, 2]
			
				# shift color measures 
				for time_index in range(0,t-1):
					black[pixel_index][time_index] = black[pixel_index][time_index+1]
					white[pixel_index][time_index] = white[pixel_index][time_index+1]
					col_mean[pixel_index][time_index] = col_mean[pixel_index][time_index+1]
					
				#reset
				white[pixel_index][t-1] = 0
				black[pixel_index][t-1] = 0

				variation = (abs(r-g) + abs(g-b) + abs(b-r))/3.0
				col_mean[pixel_index][t-1] = (r + g + b)/3.0
				if variation > -1: #< 15:
					# if j > 90:
					# 	print(current_image[i, j])
					# 	print(col_mean)
					# is it dark or pale?
					if (col_mean[pixel_index][t-1] > b_t ):
						#pale
						white[pixel_index][t-1] = 1
					else:
						if (col_mean[pixel_index][t-1] < w_t):
							black[pixel_index][t-1] = 1

				# blue
				blue_plus = 0.0
				red_plus = 0.0
				green_plus = 0.0
				# if ( (b > r) and (b > g) ):
				# 	blue_plus = b - (r+g)/2.0
				# 	# # write transitions
				# 	# white_tr = '_'.join(map(str, white[pixel_index][:4].astype(int)))
				# 	# black_tr = '_'.join(map(str, black[pixel_index][:4].astype(int)))
				# 	# row = [black_tr,white_tr]
				# 	# writer.writerow(row)
				# # red
				# else:
				# 	if ( (r > b) and (r > g ) ):
				# 		red_plus = r - (b+g)/2.0	

				# # black
				# if !((r < b_t) or (g < b_t) or (b < b_t)):
				# 	black[pixel_index][2] = 1
				# # white
				# if !((r > w_t) or (g > w_t) or (b > w_t)):
				# 	white[pixel_index][2] = 1
				
				# # blue
				# if ( ((b - blue_t) > r) and ((b - blue_t) > g )):
				# 	blue_plus = b - (r+g)/2.0
				# # red
				# else:
				# 	if (((r - red_t) > b) and ((r - red_t) > g )):
				# 		red_plus = r - (b+g)/2.0	

				col_offset = col_mean[pixel_index][t-1] - col_mean[pixel_index][t-2]
				#if(started):
				# red_plus = col_offset + r
				# green_plus = col_offset + g
				# blue_plus = col_offset + b

				global_rgb_s = rgb_surround(i,j,1,current_image)
				s_mean = sum(global_rgb_s)/3.0

				# #check time transitions
				if (black[pixel_index][0] == 1 and white[pixel_index][1] == 1
					and  s_mean > b_t ): #surround is kinda white
					bw_sum = bw_sum + 1
					red_sum[0] = red_sum[0] + r
					blue_sum[0] = blue_sum[0] + b
					new_image[i,j,0] = r
					new_image[i,j,1] = 0
					new_image[i,j,2] = b
					# for step in range(0,3):
					# 	color_tracks[0][step] = color_tracks[0][step] + previous_images[step][i,j]
					# calculate rgb of surrounding area
					# rgb_s[0] = rgb_s[0] + rgb_surround(i,j,1,current_image)


				if (white[pixel_index][0] == 1 and black[pixel_index][1] == 1
					and  s_mean > b_t ):
					wb_sum = wb_sum + 1
					red_sum[1] = red_sum[1] + r
					blue_sum[1] = blue_sum[1] + b
					new_image[i,j,0] = r
					new_image[i,j,1] = 0
					new_image[i,j,2] = b
					# for step in range(0,3):
					# 	color_tracks[1][step] = color_tracks[1][step] + previous_images[step][i,j]
					# rgb_s[1] = rgb_s[1] + rgb_surround(i,j,1,current_image)

				# if (white[pixel_index][0] == 1 
				# 	and black[pixel_index][1] == 1
				# 	and black[pixel_index][2] == 1
				# 	#and white[pixel_index][4] == 1
				# 	):
				# 	bw_sum = bw_sum + 1
				# 	red_sum[0] = red_sum[0] + red_plus
				# 	blue_sum[0] = blue_sum[0] + blue_plus
				# 	new_image[i,j,0] = r
				# 	new_image[i,j,1] = 0
				# 	new_image[i,j,2] = b
				# if (white[pixel_index][0] == 1 
				# 	and white[pixel_index][1] == 1
				# 	and white[pixel_index][2] == 1
				# 	and white[pixel_index][3] == 0
				# 	and black[pixel_index][0] == 0
				# 	and black[pixel_index][1] == 0
				# 	and black[pixel_index][2] == 0
				# 	and black[pixel_index][3] == 0
				# 	):
				# if (white[pixel_index][0] == 0 
				# 	and white[pixel_index][1] == 0
				# 	and white[pixel_index][2] == 0
				# 	and white[pixel_index][3] == 0
				# 	and black[pixel_index][0] == 1
				# 	and black[pixel_index][1] == 1
				# 	and black[pixel_index][2] == 1
				# 	and black[pixel_index][3] == 0
				# 	):
				# 	wb_sum = wb_sum + 1
				# 	red_sum[1] = red_sum[1] + red_plus
				# 	blue_sum[1] = blue_sum[1] + blue_plus
				# 	new_image[i,j,0] = r
				# 	new_image[i,j,1] = 0
				# 	new_image[i,j,2] = b

				pixel_index = pixel_index + 1

			# image_array = Image.fromarray(new_image.astype('uint8'), 'RGB')
			# name = output_dir + "/fpsi/" + image_file
			# image_array.save(name)
			# print("saved image ", name)


			# (black->white , white->black)
			# write row
			# ['bw_r','bw_b','wb_r','wb_b']
			row = [red_sum[0]/bw_sum, blue_sum[0]/bw_sum, red_sum[1]/wb_sum, blue_sum[1]/wb_sum]
			#row = []
			# color_tracks[0] = color_tracks[0]/bw_sum
			# color_tracks[1] = color_tracks[1]/wb_sum
			# row = [red_sum[0]/bw_sum, blue_sum[0]/bw_sum, red_sum[1]/wb_sum, blue_sum[1]/wb_sum]
			# row.extend(rgb_s[0]/bw_sum)
			# row.extend(rgb_s[1]/wb_sum)


			# for step in range(0,t):
			# 	row.extend(color_tracks[0][step])
			# for step in range(0,t):
			# 	row.extend(color_tracks[1][step])

			writer.writerow(row)

			started = True


# with open(output_dir+'motion_half.csv', mode='w') as csv_file:
#  	fieldnames = ['red_left', 'red_right', 'blue_left', 'blue_right']
# 	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# 	writer.writerow(fieldnames)

# 	for image_file in image_list[:image_count]:
# 		image_path = os.path.join(image_dir, image_file)
# 		print("read ", image_path)

# 		# Beware: numpy ordering is b, g, r
# 		current_image = numpy.array(Image.open(image_path).convert('RGB'))

# 		middle = current_image.shape[1]/2
# 		red, blue = motion_half(current_image, middle)

# 		n = n + 1

# 		writer.writerow([red[0], red[1], blue[0], blue[1]])



# with open(output_dir+'reds_blues.csv', mode='w') as csv_file:
# 	fieldnames = ['red', 'blue']
# 	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# 	writer.writerow(fieldnames)

# 	for image_file in image_list[:image_count]:
# 		image_path = os.path.join(image_dir, image_file)
# 		print("read ", image_path)

# 		# Beware: numpy ordering is b, g, r
# 		current_image = numpy.array(Image.open(image_path).convert('RGB'))

# 		middle = current_image.shape[0]/2
# 		result = red_blue(current_image)

# 		n = n + 1

# 		writer.writerow(result)


# with open(output_dir+'top_bottom_diff.csv', mode='w') as csv_file:
# 	fieldnames = ['red_top', 'red_bottom', 'blue_top', 'blue_bottom']
# 	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# 	writer.writerow(fieldnames)

# 	for image_file in image_list[:image_count]:
# 		image_path = os.path.join(image_dir, image_file)
# 		print("read ", image_path)

# 		# Beware: numpy ordering is b, g, r
# 		current_image = numpy.array(Image.open(image_path).convert('RGB'))

# 		middle = current_image.shape[0]/2
# 		# print(current_image.shape)
# 		# print(middle)
# 		red, blue = top_bottom_diff(current_image, middle)

# 		n = n + 1

# 		writer.writerow([red[0], red[1], blue[0], blue[1]])

# read all images but last one
# for image_file in image_list[:image_count]:
# 	image_path = os.path.join(image_dir, image_file)
# 	print("read ", image_path)

# 	# Beware: numpy ordering is b, g, r
#	# or is it
# 	current_image = numpy.array(Image.open(image_path).convert('RGB'))
# 	image_path = os.path.join(image_dir, image_list[n+1])
# 	next_image = numpy.array(Image.open(image_path).convert('RGB'))

# 	new_image = red_blue_diff(current_image, next_image)
# 	image_array = Image.fromarray(new_image.astype('uint8'), 'RGB')
# 	name = output_dir + "______" + str(n).zfill(3) + ".png"
# 	image_array.save(name)
# 	print("saved image ", name)

# 	n = n + 1


# # average red and blues by column
# image_path = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/results/" + dataset + "test_20y_0.jpg"
# current_image = numpy.array(Image.open(image_path).convert('RGB'))
# result = red_blue_green(current_image, 'colors_average_20.csv')

# # flickers
black_white_next(args.image_dir, output_dir)

# average colors on sample square
#sample_average_color(args.image_dir, output_dir)

