from abstract_image import get_random_image, get_random_image_array
from utilities import cppn
# from pytorch_neat.pytorch_neat.cpnn import create_cpnn
# from pytorch_neat.pytorch_neat.multi_env_eval import MultiEnvEvaluator
# from pytorch_neat.pytorch_neat.neat_reporter import LogReporter
# from pytorch_neat.pytorch_neat.recurrent_net import RecurrentNet
# # where is that???
# import neat
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
    return np.uint8(image)


def crossover(parents, n_offspring=1, mutation_ratio=0.1):
    print(crossover)
    # take half the pixels of each parent
    shape = parents[0].shape

    offsprings = [None]*n_offspring
    for i in range(0, n_offspring):
        # mix
        im1 = Image.fromarray(parents[0])
        im2 = Image.fromarray(parents[1])

        a = np.random.random()
        blended = Image.blend(im1, im2, alpha=a)
        #blended.save("__" + str(i) +'.png')

        # blended = parents[1].copy()
        # blended[0:shape[0]/2, 0:shape[1]/2, ] = parents[0][0:shape[0]/2, 0:shape[1]/2, ]
        # gs_min = np.zeros(shape)
        # gs_max = np.ones(shape) * 50
        # cross = np.random.sample(shape)*100
        # mask = cv2.inRange(cross, gs_min, gs_max)
        # # plt.imshow(mask)
        # # plt.show()
        # result = cv2.bitwise_and(parents[0], parents[1], mask=mask)
        # blended = Image.fromarray(result)

        # mutate
        # mutation = get_random_image_array(shape[1], shape[0])
        # im3 = Image.fromarray(mutation)
        # mutated = Image.blend(blended, im3, alpha=mutation_ratio)
        # offsprings[i] = np.array(mutated)


        # pixel mutation
        #random_pixels = generate_random_image(shape[0], shape[1])
        random_pixels = get_random_image_array(shape[1], shape[0])
        temp = Image.fromarray(random_pixels)
        #temp.show()
        a = np.random.randint(low=0, high=255, size=3)
        b =  np.random.randint(low=0, high=255, size=3)
        gs_min = a
        gs_max = b

        for j in range(3):
            if a[j] > b[j]:
                gs_min[j] = b[j]
                gs_max[j] = a[j]
       
        #print(gs_min, gs_max, random_pixels.shape)
   
        mask = cv2.inRange(random_pixels, gs_min, gs_max)
        blended = np.array(blended)
        new_pixels =  cv2.bitwise_and(random_pixels, random_pixels, mask=mask)
        maskReversed = cv2.bitwise_not(mask)
        old_pixels = cv2.bitwise_and(blended, blended, mask=maskReversed)
        mixed = cv2.bitwise_or(old_pixels, new_pixels)
        result = Image.blend(Image.fromarray(blended), Image.fromarray(mixed), alpha=0.5)

        offsprings[i] = np.array(result)
    return offsprings

def test():
    size = [160,120]
    # init_population = initial_population(shape, 2)
    init_population = [generate_random_image(size[1], size[0]), 
                        generate_random_image(size[1], size[0])]

    i = 0
    for image in init_population:
        Image.fromarray(image).save("_" + str(i) +'.png')
        i = i+1

    i = 0
    init_population = crossover(init_population, n_offspring=2)
    for image in init_population:
        Image.fromarray(image).save("__" + str(i) +'.png')
        i = i+1



def mutate_cpnn():
    # config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    # config = neat.Config(
    #     neat.DefaultGenome,
    #     neat.DefaultReproduction,
    #     neat.DefaultSpeciesSet,
    #     neat.DefaultStagnation,
    #     config_path,
    # )

    # evaluator = MultiEnvEvaluator(
    #     make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    # )

    # def eval_genomes(genomes, config):
    #     for _, genome in genomes:
    #         genome.fitness = evaluator.eval_genome(genome, config)

    # pop = neat.Population(config)
    # # stats = neat.StatisticsReporter()
    # # pop.add_reporter(stats)
    # # reporter = neat.StdOutReporter(True)
    # # pop.add_reporter(reporter)
    # # logger = LogReporter("neat.log", evaluator.eval_genome)
    # # pop.add_reporter(logger)

    # # pop.run(eval_genomes, n_generations)

    # # pop = [genome, genome,..]
    # for genome in pop:
    #     cppn_nodes = create_cppn(genome, config)
    #     print(cppn_nodes)


    n = 2
    # generate network weights
    #art = cppn.Art_Gen()
    pop = cppn.init_pop(n)
    x_res = 160
    y_res = 120
    c_dim = 3

    # get and save images
    for i in range(n):
        genome = pop[i]
        # generate input
        z = np.random.uniform(low=-1.0, high=1.0, size=(genome.batch_size, genome.h_size)).astype(np.float32)
        image_array = genome.generate(x_res, y_res,10,z)

        img_data = np.array(1-image_array)
        img_data = np.array(img_data.reshape((y_res, x_res, c_dim))*255.0, dtype=np.uint8)
        image = Image.fromarray(img_data)
        image.save("_" + str(i) +'.png')
        print("saved", i)


mutate_cpnn()
# test()
