import argparse
import numpy as np
import os
from PIL import Image


# keep only the common (resp different) points between 2 images
def save_common_points(input_path_0, input_path_1, output_dir, limit):
    input_list_0 = sorted(os.listdir(input_path_0))
    input_list_0 = sorted(os.listdir(input_path_1))
    n = len(input_list)

    for i in range(0,n):
        input_image_path_0 = input_path_0 + "/" + input_list_0[i]
        input_image_0 = np.array(Image.open(input_image_path_0).convert('RGB'))
        input_image_path_1 = input_path_1 + "/" + input_list_1[i]
        input_image_1 = np.array(Image.open(input_image_path_1).convert('RGB'))

        # take the mse for each channel and keep the mean
        mse = (np.square(input_image_0 - input_image_1)).mean(axis=None)
        

	    diff = 1.0*input_image - prediction
	    mse = (np.square(input_image - prediction)).mean(axis=None)
	    print("mse ", mse)
	    mask = (mse<limit).astype(int)
	    mean = (input_image_0 + input_image_1)/2

	    result = cv2.bitwise_and(mean, mean, mask=mask)

	    combined = np.ones(prediction.shape)
	    combined = combined*128 + result

	    image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
	    name = output_dir + "/" + str(i).zfill(10) + ".png"
	    image_array.save(name)
	    print("saved image ", name)



parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--input0', '-i0', default='', help='Path to 1st input directory')
parser.add_argument('--input1', '-i1', default='', help='Path to 2nd input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--type', '-t', type=int, default=0, help='0 for common points, 1 for differences')
parser.add_argument('--limit', '-l', type=int, default=10, help='error tolerance threshold')


args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if (args.type == 0):
	save_common_points(args.input, args.prediction, output_dir)
else:
	save_differences()
