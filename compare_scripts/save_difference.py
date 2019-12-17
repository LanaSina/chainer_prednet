import argparse
import cv2
import numpy as np
import os
from PIL import Image


# keep only the common (resp different) points between 2 images
def save_difference(input_path_0, input_path_1, output_dir, rep, off):
    w = 160

    input_list_0 = sorted(os.listdir(input_path_0))
    input_list_1 = sorted(os.listdir(input_path_1))
    n = len(input_list_0)

    if n==1:
        n = len(input_list_1)
        temp = input_list_0[0]
        input_list_0 = [temp]*n

    for i in range(off,n-off):
        if((i+1)%rep!=0):
            continue
        input_image_path_0 = input_path_0 + "/" + input_list_0[i]
        input_image_0 = np.array(Image.open(input_image_path_0).convert('RGB'))
        input_image_path_1 = input_path_1 + "/" + input_list_1[i]
        input_image_1 = np.array(Image.open(input_image_path_1).convert('RGB'))
        differences = np.zeros(input_image_0.shape).astype('uint8')

        for index in range(0,input_image_0.shape[0]):
            for j in range(0,input_image_0.shape[1]):
                if sum(input_image_1[index,j])==0:
                    for c in range(0,3):
                        differences[index,j,c] = input_image_0[index,j,c]

        image_array = Image.fromarray(differences.astype('uint8'), 'RGB')
        name = output_dir + "/" + str(i).zfill(10) + ".png"
        image_array.save(name)
        print("saved image ", name)



parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--input0', '-i0', default='', help='Path to 1st input directory')
parser.add_argument('--input1', '-i1', default='', help='Path to 2nd input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--rep', '-r', type=int, default=1, help='number of images to skip (eg 5 to skip 0..3, 5..8')
parser.add_argument('--enhance', '-e', type=int, default=0, help='save images on a grey baseline')
parser.add_argument('--offset', '-off', type=int, default=0, help='where to start on the input_0 list (eg 1 to skip 1st image)')


args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_difference(args.input0, args.input1, output_dir, args.rep, args.offset)