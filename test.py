from abstract_image import get_random_image, get_random_image_array
from utilities import cppn
from pytorch_neat.pytorch_neat.cppn import create_cppn
from pytorch_neat.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.pytorch_neat.recurrent_net import RecurrentNet
import neat
from PIL import Image
from random import random, randrange
import numpy as np
import cv2
import os

import torch



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

def get_net_image(net, x_res, y_res, z):
    c_dim = 3

    image_array = net.generate(x_res, y_res,20,z=z)
    img_data = np.array(1-image_array)
    img_data = np.array(img_data.reshape((y_res, x_res, c_dim))*255.0, dtype=np.uint8)
    image = Image.fromarray(img_data)

    return image

def neat_illusion():
    # image inputs
    # color output x,y,color
    w = 5
    h = 5
    c_dim = 3
    # x, y for each pixel
    pix = np.zeros(((w,h,2)))
    for x in range(w):
        for y in range(h):
            pix[x,y,0] = x 
            pix[x,y,1] = y 

    # pix_coord = tuple(pix)
    # print("pix_coord", pix_coord)
    # shape = torch.Size([len(pix_coord)])
    # print(shape)
    # input_ = [torch.tensor(pix_coord)]
    leaf_names = ["x","y"]
    out_names = ["r","g","b"]
  
    def eval_genomes(genomes, config):
        #print(genomes)
        for genome_id, genome in genomes:
            # [net] = create_cppn(
            #                     genome,
            #                     config,
            #                     leaf_names,
            #                     [""],
            #                 )
            # for x in range(w):
            #     for y in range(h):
            #         # s = net.activate(input_, shape)
            #         # print("s", s)
            #         image_array[x,y] = 0
                    #
            genome.fitness = np.random.random()

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "neat.cfg")

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, 3)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    delta_w_node = create_cppn(
        winner,
        config,
        leaf_names,
        out_names
    )

    # as many nodes as outputs
    print(delta_w_node) 
    image_array = np.zeros(((h,w,3)))
    r = []
    g = []
    b = []
    print("image_array[:,:,0]", image_array[:,:,0])
    # for x in range(w):
    #     for y in range(h):
    #         pix[x,y,0] = x 
    #         pix[x,y,1] = y 
    # get r g b for all x and all y
    inp_x = torch.tensor(pix[:,:,0].flatten())
    inp_y = torch.tensor(pix[:,:,1].flatten())
    c = 0
    for node in delta_w_node:
        pixel_func = delta_w_node[0]
        pixels = pixel_func(x=inp_x, y=inp_y)
        print(pixels)
        pixels_np = pixels.numpy()
        image_array[:,:,c] = np.reshape(pixels_np, (h, w))
        c = c + 1

    # delta_w = delta_w_node(_i=np.array(pix_coord))

    # print("*delta_w_node", delta_w_node)
    # print("delta_w", delta_w)

    # image_array = delta_w.numpy()
    # #delta_w_node.activate(input_, shape = shape).numpy()

    img_data = np.array(image_array*255.0, dtype=np.uint8)
    image =  Image.fromarray(img_data)
    image.save("__0.png")


def neat_cppn():

    # 2-input XOR inputs and expected outputs.
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for xi, xo in zip(xor_inputs, xor_outputs):
                output = net.activate(xi)
                genome.fitness -= (output[0] - xo[0]) ** 2

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "neat.cfg")

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, 3)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)

    print("winner", winner)
    [delta_w_node] = create_cppn(
        winner,
        config,
        ["x_in", "y_in"],
        ["delta_w"],
    )

    print("*delta_w_node", delta_w_node)
    input_ = torch.tensor(xor_inputs[1])
    result = delta_w_node.activate([input_], shape = torch.Size([2]))
    print("result", result)

    #delta_w = delta_w_node(x_in = xor_inputs[0], y_in=xor_inputs[0])


def mutate_cppn():
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

    inputs = [None]*n

    # get and save images
    for i in range(n):
        genome = pop[i]
        # generate input
        z = np.random.uniform(low=-1.0, high=1.0, size=(genome.batch_size, genome.h_size)).astype(np.float32)
        inputs[i] = z

        # image_array = genome.generate(x_res, y_res,20,z)

        # img_data = np.array(1-image_array)
        # img_data = np.array(img_data.reshape((y_res, x_res, c_dim))*255.0, dtype=np.uint8)
        # image = Image.fromarray(img_data)
        image = get_net_image(genome, x_res, y_res, z)
        image.save("_" + str(i) +'.png')

    # mutate
    p0 = pop[0]
    p1 = pop[1]
    # network weights:  self.x_dat, y_dat, r_dat
    #p0.y_dat[0,10] = p0.y_dat[0,10]*1;#.001
    image = get_net_image(p0, x_res, y_res, inputs[0])
    image.save("__0.png")
    image = get_net_image(p0, x_res, y_res, inputs[0])
    image.save("__1.png")
    # p1.x_dat[0,10] = p1.x_dat[0,10]*2
    # image = get_net_image(p1, x_res, y_res, inputs[1])
    # image.save("__1.png")




# mutate_cppn()
# neat_cppn()
neat_illusion()
# test()
