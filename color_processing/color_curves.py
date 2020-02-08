import csv
import numpy
from PIL import Image


output_dir = "results/lines_fw_black_bg_reverse/"
image_path = output_dir + "test_20y_0.png"


image = numpy.array(Image.open(image_path).convert('RGB'))
print(image.shape)
# h x w 

with open(output_dir+'average_colors.csv', mode='w') as csv_file:
	fieldnames = ['red', 'green', 'blue']
	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(fieldnames)

	#col
	for j in xrange(0,image.shape[1]):
		#row
		col_sum = [0.0,0.0,0.0] #rgb
		for i in xrange(0,image.shape[0]):
			col_sum[0] = col_sum[0] + image[i, j, 0]
			col_sum[1] = col_sum[1] + image[i, j, 1]
			col_sum[2] = col_sum[2] + image[i, j, 2]

		col_sum[0] = col_sum[0]/image.shape[1]
		col_sum[1] = col_sum[1]/image.shape[1]
		col_sum[2] = col_sum[2]/image.shape[1]

		writer.writerow(col_sum)
