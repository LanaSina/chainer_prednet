import argparse
import cv2
import csv
import numpy as np
from optical_flow.optical_flow import lucas_kanade
import os
from PIL import Image
from PredNet.call_prednet import test_prednet
from random import random, randrange
from utilities.mirror_images import mirror, TransformationType



# high score if vectors pass the mirror test
def illusion_score(vectors):
    # check vector alignements
    comp_x = 0
    comp_y = 0
    for vector in vectors:
        # dx
        comp_x = comp_x + vector[2]
        comp_y = comp_y + abs(vector[3])

    # minimize comp_y, maximize comp_x
    score = comp_x - comp_y
    return score

def random_modify(image_path):
    image = np.array(Image.open(image_path).convert('RGB'))

    w = image.shape[0]
    h = image.shape[1]
    c_range = 20

    for x in range(0,100):
        i = randrange(w)
        j = randrange(h)
        color = randrange(3)
        sign = random()

        pixel = image[i,j]
        if sign>=0.5:
            pixel[color] = pixel[color] + randrange(c_range)
            if pixel[color] > 255 : pixel[color] = 255
        else:
            pixel[color] = pixel[color] - randrange(c_range) 
            if pixel[color] < 0  : pixel[color] = 0

    return image

# take the flow vectors origins and change the pixels
def generate(input_image, output_dir, model_name):
    repeat = 1
    limit = 1
    size = [160,120]
    channels = [3,48,96,192]
    gpu = -1
    prediction_dir = output_dir + "/original/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    name = input_image.split("/")
    name = name[len(name)-1]
    temp = name.split(".")
    image = np.array(Image.open(input_image).convert('RGB'))
    image_array = Image.fromarray(image.astype('uint8'), 'RGB')
    alternate_input = output_dir + "original/" + name
    image_array.save(alternate_input)

    mirror_images_dir = output_dir + "/mirrored/"
    mirror_image = mirror_images_dir + temp[0] + ".png"
    if not os.path.exists(mirror_images_dir):
        os.makedirs(mirror_images_dir)

    images_list = [alternate_input]*repeat
    mirror_images_list = [mirror_image]*repeat
    score = 0

    i = 0
    while i < 100:
        
        # runs repeat x times on the input image, save in result folder
        test_prednet(initmodel = model_name, images_list = images_list, size=size, 
                    channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat)
        # calculate flow
        prediction_image_path = prediction_dir + str(0).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
        original_vectors = np.asarray(results["vectors"])

        #mirror image
        mirror(input_image, mirror_images_dir, True, TransformationType.MirrorAndFlip)
        
        # predict
        test_prednet(initmodel = model_name, images_list = mirror_images_list, size=size, 
                    channels = channels, gpu = gpu, output_dir = mirror_images_dir + "prediction", skip_save_frames=repeat)
        # calculate flow
        prediction_image_path = mirror_images_dir + "prediction/" + str(0).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, mirror_images_dir+"/flow/", save=True)
        mirrored_vectors = np.asarray(results["vectors"])

        # calculate score
        new_score = illusion_score(original_vectors) + illusion_score(mirrored_vectors)
        print("score", score, "new_score", new_score)
        if (score==0) or new_score>score:
            score = new_score
            i = i + 1
            image_array = Image.fromarray(image.astype('uint8'), 'RGB')
            # image_name = output_dir + "original/" + name
            image_array.save(input_image)

        # modify image
        image = random_modify(input_image)
        image_array = Image.fromarray(image.astype('uint8'), 'RGB')
        alternate_input = output_dir + "original/" + name
        image_array.save(alternate_input)
    

parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

generate(args.input, output_dir, args.model)
