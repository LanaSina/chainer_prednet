import numpy
from PIL import Image
import os


image_dir = "datasets/kitti/image_02/data"
output_dir = "image_analysis/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_list = sorted(os.listdir(image_dir))
image_count = 20 # or image_list.length
n = 0

# read all images but last one
for image_file in image_list[:image_count]:
	image_path = os.path.join(image_dir, image_file)
	print("read ", image_path)

	current_image = numpy.array(Image.open(image_path))
	next_image = numpy.array(Image.open(image_path))

	# create image with only the increases in blue or red
	new_image = numpy.zeros(current_image.shape)

	for i in xrange(0,current_image.shape[0]):
		for j in xrange(0,current_image.shape[1]):
			rdiff = next_image[i, j, 0] - current_image[i, j, 0]
			new_image[i, j, 0] = rdiff
			bdiff = next_image[i, j, 2] - current_image[i, j, 2]
			new_image[i, j, 2] = bdiff

	image_array = Image.fromarray(new_image.astype('uint8'), 'RGB')
	image_array.save(output_dir + str(n).zfill(3) + ".png")
	print("saved image ", n)

	n = n + 1
