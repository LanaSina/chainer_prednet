import argparse
import numpy as np
import os
from PIL import Image


# return an image with only overpredicted colors
def color_diff(input_image_path, prediction_path, output_dir):
    # create image with only the strongest predicted in r,g and b
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    prediction = np.array(Image.open(prediction_path).convert('RGB'))

    diff = 1.0*input_image - prediction

    plus_error = np.zeros(prediction.shape)
    minus_error = np.zeros(prediction.shape)

    # for i in range(0,prediction.shape[0]):
    #     for j in range(0,prediction.shape[1]):

    #         for c in range(0,3):
    #             if( diff[i, j, c] > 0 ):
    #                 plus_error[i, j, c] = diff[i, j, c]*2
    #             else :
    #                 minus_error[i, j, c] = -diff[i, j, c]*2
    

    combined = np.ones(prediction.shape)
    combined = combined*128 

    # save it
    # image_array = Image.fromarray(plus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_in-pre.png"
    # image_array.save(name)
    # print("saved image ", name)
    # image_array = Image.fromarray(minus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_pre-in.png"
    # image_array.save(name)
    # print("saved image ", name)

    # plus_error = combined + plus_error
    # image_array = Image.fromarray(plus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_in-pre_offset.png"
    # image_array.save(name)
    # print("saved image ", name)

    # minus_error = combined + minus_error
    # image_array = Image.fromarray(minus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_pre-in_offset.png"
    # image_array.save(name)
    # print("saved image ", name)

    combined = combined+ diff
    image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
    name = output_dir + "_combi.png"
    image_array.save(name)
    print("saved image ", name)


def save_errors(input_path, prediction_path, output_dir):
    input_list = sorted(os.listdir(input_path))
    prediction_list = sorted(os.listdir(prediction_path))
    n = len(input_list)

    for i in range(0,n):
        input_image_path = input_path + "/" + input_list[i]
        # create image with only the strongest predicted in r,g and b
        input_image = np.array(Image.open(input_image_path).convert('RGB'))
        prediction_image_path = prediction_path + "/" + prediction_list[i]
        prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

        diff = 1.0*input_image - prediction

        combined = np.ones(prediction.shape)
        combined = combined*128 

        combined = combined+ diff
        image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
        name = output_dir + "/" + prediction_list[i]
        image_array.save(name)
        print("saved image ", name)

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--input', '-i', default='', help='Path to input image or directory')
parser.add_argument('--prediction', '-p', default='', help='Path to predicted image or diectory')
parser.add_argument('--output_dir', '-d', default='', help='path of output diectory')


args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_errors(args.input, args.prediction, output_dir)
