from abstract_image import get_random_image, get_random_image_array
import argparse
import cv2
import csv
import numpy as np
from optical_flow.optical_flow import lucas_kanade
import os
from PIL import Image
from PredNet.call_prednet import test_prednet
from random import random, randrange
from utilities.mirror_images import mirror, mirror_multiple, TransformationType



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

def generate_random_image(w, h):
    image = np.random.randint(256, size=(w, h, 3))
    return image

def random_modify(image_path):
    image = np.array(Image.open(image_path).convert('RGB'))

    w = image.shape[0]
    h = image.shape[1]
    c_range = 50

    for x in range(0,500):
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


# w h 
def initial_population(img_shape, n_individuals=2):
     init_population = [None]*n_individuals

     for indv_num in range(n_individuals):
         # Randomly generating initial population chromosomes genes values.
         init_population[indv_num] = get_random_image_array(img_shape[0], img_shape[1])

     return init_population

def crossover(parents, n_offspring=1, mutation_ratio=0.1):
    # take half the pixels of each parent
    shape = parents[0].shape

    offsprings = [None]*n_offspring
    for i in range(0, n_offspring):
        # mix
        cross = np.random.sample(shape)*100
        im1 = Image.fromarray(parents[0])
        im2 = Image.fromarray(parents[1])
        blended = Image.blend(im1, im2, alpha=.5)
        #blended.save("__" + str(i) +'.png')

        # mutate
        mutation = get_random_image_array(shape[1], shape[0])
        im3 = Image.fromarray(mutation)
        mutated = Image.blend(blended, im3, alpha=mutation_ratio)
        #mutated.save("___" + str(i) +'.png')
        offsprings[i] = mutated
    return offsprings


def get_best(population, n):
    output_dir = "temp/"
    repeat = 1
    limit = 1
    size = [160,120]
    channels = [3,48,96,192]
    gpu = 0
    prediction_dir = output_dir + "/original/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    mirror_images_dir = output_dir + "mirrored/"
    if not os.path.exists(mirror_images_dir):
        os.makedirs(mirror_images_dir)

    if not os.path.exists(output_dir + "images/"):
        os.makedirs(output_dir + "images/")

    images_list = [None]* len(population)
    #save temporarily
    i = 0
    for image in population:
        image_array = Image.fromarray(image.astype('uint8'), 'RGB')
        image_name = output_dir + "images/" + str(i).zfill(10) + ".png"
        input_list[i] = image_name
        image_array.save(input_image, "PNG")
        i = i + 1

    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, images_list = images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat, reset_each = True)
    # calculate flows
    i = 0
    original_vectors = [None] * len(population)
    for input_image in images_list:
        prediction_image_path = prediction_dir + str(i).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
        original_vectors[i] = np.asarray(results["vectors"])
        i = i + 1

    #mirror images
    mirror_multiple(output_dir + "images/", mirror_images_dir, True, TransformationType.MirrorAndFlip)
    mirror_images_list = sorted(os.listdir(mirror_images_dir))
    # predict
    test_prednet(initmodel = model_name, images_list = mirror_images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = mirror_images_dir + "prediction/", skip_save_frames=repeat)
    # calculate flow
    i = 0
    mirrored_vectors = [None] * len(population)
    for input_image in mirror_images_list:
        prediction_image_path = mirror_images_dir + "prediction/" + str(i).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
        mirrored_vectors[i] = np.asarray(results["vectors"])
        i = i + 1

    # calculate score
    scores = [None] * len(population)
    for i in range(0, len(population)):
        scores[i] =[i, illusion_score(original_vectors[i]) + illusion_score(mirrored_vectors[i])]

    sorted_scores = scores.sort(key=lambda x:x[1])
    print("sorted_scores ", sorted_scores)
    results = [population[i] for i in sorted_scores[0:n]]

    return results

# take the flow vectors origins and change the pixels
def generate(input_image, output_dir, model_name):
    repeat = 1
    limit = 1
    size = [160,120]
    channels = [3,48,96,192]
    gpu = 0
    prediction_dir = output_dir + "/original/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    name = input_image.split("/")
    name = name[len(name)-1]
    temp = name.split(".")
    alternate_input = output_dir + "original/" + name
    # image = np.array(Image.open(alternate_input).convert('RGB'))
    image = get_random_image(size[1], size[0]) #np.array(Image.open(input_image).convert('RGB'))
    image_array = Image.fromarray(image.astype('uint8'), 'RGB')
    image_array.save(alternate_input)

    mirror_images_dir = output_dir + "/mirrored/"
    mirror_image = mirror_images_dir + temp[0] + ".png"
    if not os.path.exists(mirror_images_dir):
        os.makedirs(mirror_images_dir)

    images_list = [alternate_input]*repeat
    mirror_images_list = [mirror_image]*repeat
    score = 0

    size = 2
    init_population = initial_population[size]
    # Generating next generation using crossover.
    new_population = crossover(init_population, n_offspring=2)
    best = get_best(new_population, 2)
    i = 0
    for image_array in best:
        image = Image.fromarray(image_array)
        image.save(output_dir + "/best/" + str(i).zfill(10) +'.png')
        i = i+1

    # add parents
    next_population = init_population
    next_population.extend(best)

    for i in range(0,500):
        best = get_best(next_population, 2)
        im = 0
        for image_array in best:
            image = Image.fromarray(image_array)
            image.save(output_dir + "/best/" + str(im).zfill(10) +'.png')
            im = im+1
        new_population = crossover(best, n_offspring=2)
        
        # add parents
        next_population.extend(best)



    # while score < 100:        
        
    #     # runs repeat x times on the input image, save in result folder
    #     test_prednet(initmodel = model_name, images_list = images_list, size=size, 
    #                 channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat)
    #     # calculate flow
    #     prediction_image_path = prediction_dir + str(0).zfill(10) + ".png"
    #     results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
    #     original_vectors = np.asarray(results["vectors"])

    #     #mirror image
    #     mirror(input_image, mirror_images_dir, True, TransformationType.MirrorAndFlip)
        
    #     # predict
    #     test_prednet(initmodel = model_name, images_list = mirror_images_list, size=size, 
    #                 channels = channels, gpu = gpu, output_dir = mirror_images_dir + "prediction", skip_save_frames=repeat)
    #     # calculate flow
    #     prediction_image_path = mirror_images_dir + "prediction/" + str(0).zfill(10) + ".png"
    #     results = lucas_kanade(input_image, prediction_image_path, mirror_images_dir+"/flow/", save=True)
    #     mirrored_vectors = np.asarray(results["vectors"])

    #     # calculate score
    #     new_score = illusion_score(original_vectors) + illusion_score(mirrored_vectors)
    #     print("score", score, "new_score", new_score)
    #     if (score==0) or new_score>score:
    #         score = new_score
    #         image_array = Image.fromarray(image.astype('uint8'), 'RGB')
    #         # image_name = output_dir + "original/" + name
    #         image_array.save(input_image)

    #     # modify image
    #     image = random_modify(input_image)
    #     image_array = Image.fromarray(image.astype('uint8'), 'RGB')
    #     alternate_input = output_dir + "original/" + name
    #     image_array.save(alternate_input)
    


parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

generate(args.input, output_dir, args.model)

