import argparse
import cv2
import numpy as np
import os
from PIL import Image


# keep only the common (resp different) points between 2 images
def save_common_points(input_path_0, input_path_1, output_dir, limit, h0, h1, rep, off, enhance):
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

        if enhance:
            combined = np.ones(input_image_0.shape)
            combined = combined*128
        else :
            combined = np.zeros(input_image_0.shape)

        #take halves
        if (h0==0):
            input_image_0 = input_image_0[:,0:int(w/2),:] 
        else:
            input_image_0 = input_image_0[:,int(w/2):w,:] 
        if (h1==0):
            input_image_1 = input_image_1[:,0:int(w/2),:] 
        else:
            input_image_1 = input_image_1[:,int(w/2):w,:] 

        # take the mse for each channel and keep the mean
        # mse = np.square(input_image_0 - input_image_1)/2
        # print("mse ", mse)
        # mask = (mse<limit).astype(np.int8)
        mean = (input_image_0*1.0 + input_image_1*1.0)/2
        mean = mean.astype(int)

        mse = (np.square(input_image_0 - input_image_1)).mean(axis=2)
        mask = (mse<limit).astype(np.int8)
        #print(mask)
        #mask = cv2.inRange(mse, 0, limit)
        result = cv2.bitwise_and(mean, mean, mask=mask)
        if enhance:
            combined = combined + result*0.5
        else:
            combined[:,0:int(w/2),:] = result

        # no faster way
        # for index in range(0,input_image_0.shape[0]):
        #     for j in range(0,input_image_0.shape[1]):
        #         for c in range(0,3):
        #             if mask[index,j,c] and (input_image_1[index,j,c]>0):
        #                 if enhance:
        #                     combined[index,j,c] = combined[index,j,c] + (input_image_0[index,j,c]*0.5)
        #                 else:
        #                     combined[index,j,c] = combined[index,j,c] + mean[index,j,c]

        image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
        name = output_dir + "/" + str(i).zfill(10) + ".png"
        image_array.save(name)
        print("saved image ", name)


def save_differences(input_path_0, input_path_1, output_dir, limit, h0, h1, rep, off, enhance):
    w = 160

    input_list_0 = sorted(os.listdir(input_path_0))
    input_list_1 = sorted(os.listdir(input_path_1))
    n = len(input_list_0)

    if n==1:
        n = len(input_list_1)
        temp = input_list_0[0]
        input_list_0 = [temp]*n
        print(input_list_0)

    for i in range(off,n-off):
        if((i+1)%rep!=0):
            continue
        input_image_path_0 = input_path_0 + "/" + input_list_0[i]
        input_image_0 = np.array(Image.open(input_image_path_0).convert('RGB'))
        input_image_path_1 = input_path_1 + "/" + input_list_1[i]
        input_image_1 = np.array(Image.open(input_image_path_1).convert('RGB'))

        if enhance:
            combined = np.ones(input_image_0.shape)
            combined = combined*128
        else:
            combined = np.zeros(input_image_0.shape)

        #take halves
        if (h0==0):
            input_image_0 = input_image_0[:,0:int(w/2),:] 
        else:
            input_image_0 = input_image_0[:,int(w/2):w,:] 
        if (h1==0):
            input_image_1 = input_image_1[:,0:int(w/2),:] 
        else:
            input_image_1 = input_image_1[:,int(w/2):w,:] 

        # take the mse for each channel and keep the mean
        # mse = np.square(input_image_0 - input_image_1)/2
        # mask = (mse>limit).astype(np.int8)
        mean = (input_image_0*1.0 + input_image_1*1.0)/2
        mean = mean.astype(int)

        mse = (np.square(input_image_0 - input_image_1)).mean(axis=2)
        mask = (mse>limit).astype(np.int8)
        # print(mask)
        # mask = cv2.inRange(mse, 0, limit)
        result = cv2.bitwise_and(input_image_0, input_image_0, mask=mask)
        if enhance:
            combined = combined + result*0.5
        else:
            combined[:,0:int(w/2),:] = result

        # no faster way
        # for index in range(0,input_image_0.shape[0]):
        #     for j in range(0,input_image_0.shape[1]):
        #         for c in range(0,3):
        #             if mask[index,j,c] and (input_image_1[index,j,c]>0):
        #                 if enhance:
        #                     combined[index,j,c] = combined[index,j,c] + (input_image_0[index,j,c]*0.5)
        #                 else:
        #                     combined[index,j,c] = input_image_0[index,j,c]

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
parser.add_argument('--half0', '-h0', type=int, default=0, help='which half to use (0 for left, 1 for right)')
parser.add_argument('--half1', '-h1', type=int, default=0, help='which half to use (0 for left, 1 for right)')
parser.add_argument('--rep', '-r', type=int, default=1, help='number of images to skip (eg 5 to skip 0..3, 5..8')
parser.add_argument('--enhance', '-e', type=int, default=0, help='1 save images on a grey baseline')
parser.add_argument('--offset', '-off', type=int, default=0, help='where to start on the input_0 list (eg 1 to skip 1st image)')


args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if (args.type == 0):
    save_common_points(args.input0, args.input1, output_dir, args.limit, args.half0, args.half1, args.rep, args.offset, args.enhance)
else:
    save_differences(args.input0, args.input1, output_dir, args.limit, args.half0, args.half1, args.rep,  args.offset, args.enhance)