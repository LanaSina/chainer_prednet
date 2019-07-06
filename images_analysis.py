import csv
import numpy
from PIL import Image
import os


image_dir = "datasets/kitti/image_02/data"
output_dir = "image_analysis/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_list = sorted(os.listdir(image_dir))
image_count = 80 # or image_list.length
n = 0

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


def red_blue_diff(current_image, next_image):
	# create image with only the increases in blue or red
	new_image = numpy.ones(current_image.shape)
	# white background
	new_image = new_image*255

	for i in xrange(0,current_image.shape[0]):
		for j in xrange(0,current_image.shape[1]):

			rdiff = next_image[i, j, 0] - current_image[i, j, 0]
			bdiff = next_image[i, j, 2] - current_image[i, j, 2]

			lim = 100
			if (bdiff<lim and rdiff>lim):
				# new_image[i, j, 0] = rdiff
				new_image[i, j, 1] = 0
				new_image[i, j, 2] = 0
			if (rdiff<lim and bdiff>lim):
				new_image[i, j, 0] = 0
				new_image[i, j, 1] = 0
				# new_image[i, j, 2] = bdiff
	return new_image

with open(output_dir+'top_bottom_diff.csv', mode='w') as csv_file:
	fieldnames = ['red_top', 'red_bottom', 'blue_top', 'blue_bottom']
	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(fieldnames)

	for image_file in image_list[:image_count]:
		image_path = os.path.join(image_dir, image_file)
		print("read ", image_path)

		# Beware: numpy ordering is b, g, r
		current_image = numpy.array(Image.open(image_path).convert('RGB'))

		middle = current_image.shape[0]/2
		# print(current_image.shape)
		# print(middle)
		red, blue = top_bottom_diff(current_image, middle)

		n = n + 1

		writer.writerow([red[0], red[1], blue[0], blue[1]])

# # read all images but last one
# for image_file in image_list[:image_count]:
# 	image_path = os.path.join(image_dir, image_file)
# 	print("read ", image_path)

# 	# Beware: numpy ordering is b, g, r
# 	current_image = numpy.array(Image.open(image_path).convert('RGB'))
# 	image_path = os.path.join(image_dir, image_list[n+1])
# 	next_image = numpy.array(Image.open(image_path).convert('RGB'))

# 	new_image = red_blue_diff()
# 	image_array = Image.fromarray(new_image.astype('uint8'), 'RGB')
# 	name = output_dir + str(n).zfill(3) + ".png"
# 	image_array.save(name)
# 	print("saved image ", name)

# 	middle = current_image.shape[0]/2
# 	print(current_image.shape)
# 	print(middle)
# 	rb = top_bottom_diff(current_image, middle)

# 	n = n + 1

