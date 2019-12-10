import argparse
import cv2
import numpy as np
import os
from PIL import Image


# keep only the common (resp different) points between 2 images
def save_common_points(input_path_0, input_path_1, output_dir, limit):
    input_list_0 = sorted(os.listdir(input_path_0))
    input_list_1 = sorted(os.listdir(input_path_1))
    n = len(input_list_0)

    for i in range(0,n):
        input_image_path_0 = input_path_0 + "/" + input_list_0[i]
        input_image_0 = np.array(Image.open(input_image_path_0).convert('RGB'))
        input_image_path_1 = input_path_1 + "/" + input_list_1[i]
        input_image_1 = np.array(Image.open(input_image_path_1).convert('RGB'))

        combined = np.ones(input_image_0.shape)
        combined = combined*128

        # take the mse for each channel and keep the mean
        mse = np.square(input_image_0 - input_image_1)/2
        # print("mse ", mse)
        mask = (mse<limit).astype(np.int8)
        mean = (input_image_0 + input_image_1)/2
        mean = mean.astype(int)

        # result = cv2.bitwise_and(mean, mean, mask=mask)
        # no faster way
        for i in range(0,input_image_0.shape[0]):
            for j in range(0,input_image_0.shape[1]):
                for c in range(0,3):
                    if mask[i,j,c]:
                        combined[i,j,c] = combined[i,j,c] + mean[i,j,c]

        image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
        name = output_dir + "/" + str(i).zfill(10) + ".png"
        image_array.save(name)
        print("saved image ", name)



parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--input0', '-i0', default='', help='Path to 1st input directory')
parser.add_argument('--input1', '-i1', default='', help='Path to 2nd input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--type', '-t', type=int, default=0, help='0 for common points, 1 for differences, 2 for image halves')
parser.add_argument('--limit', '-l', type=int, default=10, help='error tolerance threshold')


args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if (args.type == 0):
    save_common_points(args.input0, args.input1, output_dir, args.limit)
else:
    save_differences()
