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
from optical_flow.optical_flow import lucas_kanade




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

    image_array = net.generate(x_res, y_res,1,z=z)
    img_data = np.array(1-image_array)
    img_data = np.array(img_data.reshape((y_res, x_res, c_dim))*255.0, dtype=np.uint8)
    image = Image.fromarray(img_data)

    return image

def create_grid(x_res = 32, y_res = 32, scaling = 1.0):

    num_points = x_res*y_res
    x_range = np.linspace(-1*scaling, scaling, num = x_res)
    y_range = np.linspace(-1*scaling, scaling, num = y_res)
    x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))
    y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), 1).reshape(1, num_points, 1)
    y_mat = np.tile(y_mat.flatten(), 1).reshape(1, num_points, 1)
    r_mat = np.tile(r_mat.flatten(), 1).reshape(1, num_points, 1)

    return x_mat, y_mat, r_mat

def neat_illusion():
    # image inputs
    # color output x,y,color
    w = 160
    h = 120
    c_dim = 3
    scaling = 2
    # x, y for each pixel
    pix = np.zeros(((w,h,3)))
    center = [w/2, h/2]
    for x in range(w):
        for y in range(h):
            pix[x,y,0] = x 
            pix[x,y,1] = y 
            pix[x,y,2] = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1])

    # space vector

    leaf_names = ["x","y","r"]
    out_names = ["r","g","b"]
  
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
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
    winner = p.run(eval_genomes, 20)

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
    # print(delta_w_node) 
    image_array = np.zeros(((w,h,3)))
  
    x_dat, y_dat, r_dat = create_grid(w, h, scaling)
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())
    inp_r = torch.tensor(r_dat.flatten())
   
    c = 0
    for node_func in delta_w_node:
        pixels = node_func(x=inp_x, y=inp_y, r = inp_r)
        pixels_np = pixels.numpy()
        image_array[:,:,c] = np.reshape(pixels_np, (w, h))
        c = c + 1

    img_data = np.array(image_array*255.0, dtype=np.uint8)
    image =  Image.fromarray(np.reshape(img_data,(h,w,c_dim)))
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
                         "./neat.cfg")

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
    # p0 = pop[0]
    # p1 = pop[1]
    # image = get_net_image(p0, x_res, y_res, inputs[0])
    # image.save("__0.png")
    # image = get_net_image(p0, x_res, y_res, inputs[0])
    # image.save("__1.png")


def fully_connected(input, out_dim, with_bias = True, mat = None):
    if mat is None:
        mat = np.random.standard_normal(size = (input.shape[1], out_dim)).astype(np.float32)

    result = np.matmul(input, mat)

    if with_bias == True:
        bias = np.random.standard_normal(size =(1, out_dim)).astype(np.float32)
        result += bias * np.ones((input.shape[0], 1), dtype = np.float32)

    return result

def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-1* x))  

def soft_plus(x):

    return np.log(1.0 + np.exp(x))  


def raw_test():
    w = 160
    h = 120
    scaling = 10
    num_points = w*h
    c_dim = 3
    net_size = 32
    num_layers = 3
    h_size = 32
    # x, y for each pixel
    # pix = np.zeros(((w,h,3)))
    # center = [w/2, h/2]
    # for x in range(w):
    #     for y in range(h):
    #         pix[x,y,0] = x 
    #         pix[x,y,1] = y 
    #         pix[x,y,2] = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1])

    # # get r g b for all x and all y
    # inp_x = np.reshape(pix[:,:,0].flatten(), (w*h,1))
    # inp_y = np.reshape(pix[:,:,1].flatten(), (w*h,1))
    # inp_r = np.reshape(pix[:,:,2].flatten(), (w*h,1))


    x_dat, y_dat, r_dat = create_grid(w, h, scaling)
    # Unwrap the grid matrices      
    x_dat_unwrapped = np.reshape(x_dat, (num_points, 1))
    y_dat_unwrapped = np.reshape(y_dat, (num_points, 1))
    r_dat_unwrapped = np.reshape(r_dat, (num_points, 1))

    # space vector
    z = np.random.uniform(low=-1.0, high=1.0, size=(1, h_size)).astype(np.float32)
    # hid_vec_scaled = np.reshape(z, (1, 1, h_size)) * \
    #                         np.ones((num_points, 1), dtype = np.float32)

    # hidden vector (?)
    # hid_vec = np.random.standard_normal(size =  (1, h_size))
    hid_vec_scaled = np.reshape(z, (1, 1, h_size)) * \
                     np.ones((num_points, 1), dtype = np.float32) * scaling
    h_vec_unwrapped = np.reshape(hid_vec_scaled, (num_points, h_size))


    art_net = fully_connected(h_vec_unwrapped, net_size) + \
              fully_connected(x_dat_unwrapped, net_size, with_bias = False) + \
              fully_connected(y_dat_unwrapped, net_size, with_bias = False) + \
              fully_connected(r_dat_unwrapped, net_size, with_bias = False)

    hidden = np.tanh(art_net)
    for i in range(num_layers):
        hidden = np.tanh(fully_connected(hidden, net_size, True))
    out = sigmoid(fully_connected(hidden, c_dim, True))
    print(out.shape)

    model = np.reshape(out, (w, h, c_dim))
    img_data = np.array(model*255.0, dtype=np.uint8)
    image =  Image.fromarray(np.reshape(img_data,(h,w,c_dim)))
    image.save("_1.png")

    # art = cppn.Art_Gen()
    # art.initialise_CPPN(1, net_size, h_size, RGB = True)
    # image = get_net_image(art, w, h, z)
    # image.save("_0.png")

def circle_tangent(vectors):
    w = 160
    h = 120
    c = [w/2.0, h/2.0]
    mean_ratio = 0
    # if beta = angle between radius and current vector
    # ratio of projection of V on tangent / ||V|| = sin(beta)
    # ratio = sin(arcos(R*V/||V||*||R||)) = sqrt(1- a^2)
    for v in vectors:
        # radius vector R from image center to origin of V
        r = [c[0], c[1], v[0]-c[0], v[1]-c[1]]
        # a = V*R / ||V||*||R||
        a = r[2] * v[2] + r[3] * v[3]
        a = a/(np.sqrt(r[2]*r[2] + r[3]*r[3]) * np.sqrt(v[2]*v[2] + v[3]*v[3]) )
        #print("a", a)
        # ratio
        ratio = np.sqrt(1 - a*a)
        mean_ratio = mean_ratio + ratio
        # this would be negative if going counter clockwise
        print("ratio", ratio)

    mean_ratio = mean_ratio/len(vectors)
    return mean_ratio

def rotation_ratio():
    input_image = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/datasets/snake/snake.jpg"
    # calculate flows
    prediction_image_path = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/results/fpsi_50/snake/0000000006.png"
    #"/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/results/fpsi_50/snake_composite/0000000009.png"
    #"/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/results/fpsi_50/snake/0000000006.png"
    results = lucas_kanade(input_image, prediction_image_path, "./test", save=True)
    mean_ratio = circle_tangent(results["vectors"])
    print("mean_ratio", mean_ratio)


#rotation_ratio()
mutate_cppn()
# neat_cppn()
# neat_illusion()
#raw_test()
