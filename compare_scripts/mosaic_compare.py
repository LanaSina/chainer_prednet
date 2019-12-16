import argparse
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# keep only the common (resp different) points between 2 images
def save_common_points(input_path_0, input_path_1, output_dir, limit, rep, off, enhance):
    w = 160
    h = 120
    dw = 5
    dh = 4

    x_div = int(w/dw)
    y_div = 1*int(h/dh)

    input_list_0 = sorted(os.listdir(input_path_0))
    input_list_1 = sorted(os.listdir(input_path_1))
    n = len(input_list_0)

    if n==1:
        n = len(input_list_1)
        temp = input_list_0[0]
        input_list_0 = [temp]*n

    for i in range(0,n):
        if((i+1)%rep!=0):
            continue
        input_image_path_0 = input_path_0 + "/" + input_list_0[i]
        input_image_0 = np.array(Image.open(input_image_path_0).convert('RGB'))
        input_image_path_1 = input_path_1 + "/" + input_list_1[i+off]
        input_image_1 = np.array(Image.open(input_image_path_1).convert('RGB'))

        if enhance:
            combined = np.ones(input_image_0.shape)
            combined = combined*128
        else :
            combined = np.zeros(input_image_0.shape)

        # compare each section of the input_0 mosaic with each section of the input_1 mosaic
        for x in range(0,dw):
            xstart = x*x_div
            xend = (x+1)*x_div
            for y in range(0,dh):
                ystart = y*y_div
                yend = (y+1)*y_div
                print(xstart, xend, ystart, yend)
                # part of input 0
                p_0 = input_image_0[ystart:yend,xstart:xend:]
                mean = p_0*1.0
                # avoid empty pixels
                count = (p_0.mean(axis=2)>0).astype(np.int8)
                mse = np.zeros(p_0.shape)

                for xx in range(0,dw):
                    xxstart = xx*x_div
                    xxend = (xx+1)*x_div
                    for yy in range(0,dh):
                        yystart = yy*y_div
                        yyend = (yy+1)*y_div
                        print(xxstart, xxend, yystart, yyend)
                        # part of input 1
                        p_1 = input_image_1[yystart:yyend,xxstart:xxend:]
                        # test = abs(p_0-p_1)
                        # fig=plt.figure(figsize=(8, 8))
                        # columns = 1
                        # rows = 3
                        # fig.add_subplot(rows, columns, 1)
                        # plt.imshow(p_0)
                        # fig.add_subplot(rows, columns, 2)
                        # plt.imshow(p_1)
                        # fig.add_subplot(rows, columns, 3)
                        # plt.imshow(test)
                        # plt.show()

                        # plt.imshow(test)
                        # plt.show()
                        # compare both
                        # take the mse for all channels and keep the mean
                        mean = mean + p_1*1.0
                        count = count + (p_1.mean(axis=2)>0).astype(np.int8)
                        mse = mse + 1.0*np.square(p_0 - p_1)
                        
                mse = (mse/(dh*dw)).mean(axis=2)
                mean[:,:,0] = mean[:,:,0]/count
                mean[:,:,1] = mean[:,:,1]/count
                mean[:,:,2] = mean[:,:,2]/count
                mean = mean.astype(int)

                mask = (mse<limit).astype(np.int8)
                result = cv2.bitwise_and(mean, mean, mask=mask)
                if enhance:
                    combined[ystart:yend,xstart:xend:] = combined[ystart:yend,xstart:xend:] + result*0.5
                else:
                    combined[ystart:yend,xstart:xend:] = result

        # compare rgb separately
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


def save_differences(input_path_0, input_path_1, output_dir, limit, rep, off, enhance):
    w = 160
    h = 120
    dw = 5
    dh = 4

    x_div = int(w/dw)
    y_div = 1*int(h/dh)

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

        # compare each section of the input_0 mosaic with each section of the input_1 mosaic
        for x in range(0,dw):
            xstart = x*x_div
            xend = (x+1)*x_div
            for y in range(0,dh):
                ystart = y*y_div
                yend = (y+1)*y_div
                # part of input 0
                p_0 = input_image_0[ystart:yend,xstart:xend:]
                mse = np.zeros(p_0.shape)

                for xx in range(0,dw):
                    xxstart = xx*x_div
                    xxend = (xx+1)*x_div
                    for yy in range(0,dh):
                        yystart = yy*y_div
                        yyend = (yy+1)*y_div
                        # part of input 1
                        p_1 = input_image_1[yystart:yyend,xxstart:xxend:]
                        # compare both
                        # take the mse for all channels and keep the mean
                        mse = mse + 1.0*np.square(p_0 - p_1)

                mse = (mse/(dh*dw)).mean(axis=2)
                mask = (mse>limit).astype(np.int8)
                result = cv2.bitwise_and(p_0, p_0, mask=mask)
                if enhance:
                    combined[ystart:yend,xstart:xend:] = combined[ystart:yend,xstart:xend:] + result*0.5
                else:
                    combined[ystart:yend,xstart:xend:] = result
                        
                mse = (np.square(input_image_0 - input_image_1)).mean(axis=2)
                mask = (mse>limit).astype(np.int8)

        # rgb
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
parser.add_argument('--rep', '-r', type=int, default=1, help='number of images to skip (eg 5 to skip 0..3, 5..8')
parser.add_argument('--enhance', '-e', type=int, default=0, help='1 save images on a grey baseline')
parser.add_argument('--offset', '-off', type=int, default=0, help='where to start on the input_0 list (eg 1 to skip 1st image)')


args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if (args.type == 0):
    save_common_points(args.input0, args.input1, output_dir, args.limit, args.rep, args.offset, args.enhance)
else:
    save_differences(args.input0, args.input1, output_dir, args.limit, args.rep,  args.offset, args.enhance)
