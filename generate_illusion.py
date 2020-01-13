import argparse
import cv2
import csv
from utilities.mirror_images import mirror, TransformationType
import os
from PredNet.call_prednet import test_prednet


# high score if vectors pass the mirror test
def calculate_flow_score(vectors):
    pass

# take the flow vectors origins and change the pixels
def generate(input_image, output_dir, model_name):
    repeat = 10
    limit = 1
    size = [160,120]
    channels = [3,48,96,192]
    gpu = -1
    prediction_dir = output_dir + "/original/prediction/"

    images_list = [input_image]*repeat
    print(images_list)

    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, images_list = images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat)
    # calculate flow
    prediction_image_path = prediction_dir + str(0).zfill(10) + ".png"
    results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=true)
    original_vectors = np.asarray(results["vectors"])

    #mirror image
    mirror_images_dir = output_dir + "/mirrored/"
    if not os.path.exists(mirror_images_dir):
        os.makedirs(mirror_images_dir)
    mirror(input_image, mirror_images_dir, True, TransformationType.MirrorAndFlip)
    name = input_image.split("/")
    name = name[len(name)-1]
    temp = name.split(".")
    mirror_image = mirror_images_dir + temp[0] + ".png"
    # predict
    images_list = [mirror_image]*repeat
    test_prednet(initmodel = model_name, images_list = images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = mirror_images_dir + "prediction", skip_save_frames=repeat)
    # calculate flow
    prediction_image_path = mirror_images_dir + prediction_dir + str(0).zfill(10) + ".png"
    results = lucas_kanade(input_image, prediction_image_path, mirror_images_dir+"/mirrored/flow/", save=true)
    mirrored_vectors = np.asarray(results["vectors"])

    # calculate score


    # modify image

    # repeat


    

parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

generate(args.input, output_dir, args.model)
