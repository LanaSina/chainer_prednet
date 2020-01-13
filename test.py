from abstract_image import get_random_image, get_random_image_array
from PIL import Image
from random import random, randrange
import numpy as np
import cv2


# w h 
def initial_population(img_shape, n_individuals=2):
     init_population = [None]*n_individuals

     for indv_num in range(n_individuals):
         # Randomly generating initial population chromosomes genes values.
         init_population[indv_num] = get_random_image_array(img_shape[0], img_shape[1])

     return init_population

def generate_random_image(w, h):
    image = np.random.randint(256, size=(w, h, 3))
    return image


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

def test():
    shape = [160,120]
    init_population = initial_population(shape, 2)

    i = 0
    for image in init_population:
        Image.fromarray(image).save("_" + str(i) +'.png')
        i = i+1

    new_population = crossover(init_population, n_offspring=2)
    


test()
