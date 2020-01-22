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

from pytorch_neat.pytorch_neat.cppn import create_cppn
from pytorch_neat.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.pytorch_neat.recurrent_net import RecurrentNet
import neat
import torch


# high score if vectors pass the mirror test
def illusion_score(vectors, flipped=False, mirrored=False):
    # check vector alignements
    comp_x = 0
    count = 0
    for vector in vectors:
        # normalize
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])

        # print("norm", norm)
        if norm> 0.15 or norm==0: 
            continue

        if mirrored:
            comp_x = comp_x + (-vector[2]/norm)
        else:
            comp_x = comp_x + vector[2]/norm
        #comp_y = comp_y + abs(vector[3])/norm

    # minimize comp_y, maximize comp_x
    score = comp_x
    return score

# returns ratio and vectors that are not unplausibly big
def plausibility_ratio(vectors):
    r = []
    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15: # or norm==0: 
            continue
        r.append(vector)

    ratio = len(r)/len(vectors)
    return [ratio, r]

#returns mean of vectors norms
def strength_number(vectors):
    sum_v = 0
    total_v = 0

    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        sum_v = sum_v + norm
        total_v = total_v +1
    
    return sum_v/total_v

# returns the mirroring score (lower == better) 
def mirroring_score(vectors, m_vectors):
    # print("vectors", vectors)
    sum_v = [0,0]
    for vector in vectors:
        sum_v = [sum_v[0] + vector[2], sum_v[1] + vector[3]]

    sum_mv = [0,0]
    for vector in m_vectors:
        sum_mv = [sum_mv[0] + vector[2], sum_mv[1] + vector[3]]

    s0x = sum_v[0] + sum_mv[0]
    s0y = sum_v[1] + sum_mv[1]

    return abs(s0x) + abs(s0y)

# return the mirrored score on x and y, 
# the global strength of all plausible vectors, 
# and the ratio of plausible vectors vs too big vectors
def combined_illusion_score(vectors, m_vectors):
    # check vector alignements
    sum_v = [0,0]
    total_v = 0
    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15 or norm==0: 
            continue
        sum_v = [sum_v[0] + vector[2], sum_v[1] + vector[3]]
        total_v = total_v +1

    sum_mv = [0,0]
    total_mv = 0
    for vector in m_vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15 or norm==0: 
            continue
        sum_mv = [sum_mv[0] + vector[2], sum_mv[1] + vector[3]]
        total_mv = total_mv +1

    s0x = sum_v[0] + sum_mv[0]
    s0y = sum_v[1] + sum_mv[1]
    s1 = abs(sum_v[0]) +  abs(sum_v[1]) +  abs(sum_mv[0]) +  abs(sum_mv[1])
    s2 = total_v + total_mv
    if s2 == 0:
        s2 = 0.01
    else:
        s2 = s2 / (len(vectors) + len(m_vectors))

    return [s0x + s0y, s1, s2]

# returns a high score if vectors are aligned on concentric circles
# [ratio of tangent, sum of directions]
def circle_tangent_ratio(vectors, limits = None):
    w = 160
    h = 120
    c = [w/2.0, h/2.0]
    mean_ratio = 0
    global_sum = [0,0]
    abs_sum = [0,0]
    # if beta = angle between radius and current vector
    # ratio of projection of V on tangent / ||V|| = sin(beta)
    # ratio = sin(arcos(R*V/||V||*||R||)) = sqrt(1- a^2)
    count = 0
    for v in vectors:
        # radius vector R from image center to origin of V
        r = [c[0], c[1], v[0]-c[0], v[1]-c[1]]
        norm_r = np.sqrt(r[2]*r[2] + r[3]*r[3])
        norm_v = np.sqrt(v[2]*v[2] + v[3]*v[3])
        if not limits is None:
            if (norm_r<limits[0]) or (norm_r>limits[1]):
                continue

        global_sum = global_sum + v[2:3]
        abs_sum = abs_sum + abs(v[2:3])

        # projection of vectors on each other a = V*R / ||V||*||R||
        a = r[2] * v[2] + r[3] * v[3]
        a = a/(norm_r * norm_v)
        # need the sign of the angle for orientation of vector
        if(a>0):
            # ratio
            ratio = np.sqrt(1 - a*a)
            mean_ratio = mean_ratio + ratio
        count = count + 1

    if count > 0:
        mean_ratio = mean_ratio/count
    else:
        mean_ratio = 0

    s_sum = np.sqrt(global_sum[0]*global_sum[0] + global_sum[1]*global_sum[1])
    s_sum = s_sum / np.sqrt(abs_sum[0]*abs_sum[0] + abs_sum[1]*abs_sum[1])

    return [mean_ratio,s_sum]


def generate_random_image(w, h):
    image = np.random.randint(256, size=(w, h, 3))
    return np.uint8(image)

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

def fully_connected(input, out_dim, with_bias = True, mat = None):
    if mat is None:
        mat = np.random.standard_normal(size = (input.shape[1], out_dim)).astype(np.float32)

    result = np.matmul(input, mat)

    if with_bias == True:
        bias = np.random.standard_normal(size =(1, out_dim)).astype(np.float32)
        result += bias * np.ones((input.shape[0], 1), dtype = np.float32)

    return result

def get_fidelity(input_image_path, prediction_image_path):
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

    err = np.sum((input_image.astype("float") - prediction.astype("float")) ** 2)
    err /= (float(input_image.shape[0] * input_image.shape[1])*255*255)
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return 1-err


# population:  [id, net]
def get_fitnesses_neat(population, model_name, config, id=0):
    print("fitnesses of ", len(population))
    output_dir = "temp" + str(id) + "/"
    repeat = 10
    w = 160
    h = 120
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0
    c_dim = 3
    scaling = 10

    prediction_dir = output_dir + "/original/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    mirror_dir = output_dir + "mirrored/"
    if not os.path.exists(mirror_dir+ "flow/"):
        os.makedirs(mirror_dir +"flow/")
    mirror_images_dir = mirror_dir+ "images/"
    if not os.path.exists(mirror_images_dir):
        os.makedirs(mirror_images_dir)

    if not os.path.exists(output_dir + "images/"):
        os.makedirs(output_dir + "images/")

    images_list = [None]* len(population)
    repeated_images_list = [None]* (len(population) + repeat)
    #save temporarily
    leaf_names = ["x","y","r"]
    out_names = ["r","g","b"]
    x_dat, y_dat, r_dat = create_grid(w, h, scaling)
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())
    inp_r = torch.tensor(r_dat.flatten())

    i = 0
    for genome_id, genome in population:
        image_array = np.zeros(((w,h,3)))
        c = 0
        net_nodes = create_cppn(
            genome,
            config,
            leaf_names,
            out_names
        )
        for node_func in net_nodes:
            pixels = node_func(x=inp_x, y=inp_y, r = inp_r)
            pixels_np = pixels.numpy()
            image_array[:,:,c] = np.reshape(pixels_np, (w, h))
            c = c + 1

        img_data = np.array(image_array*255.0, dtype=np.uint8)
        image =  Image.fromarray(np.reshape(img_data,(h,w,c_dim)))
        image_name = output_dir + "images/" + str(i).zfill(10) + ".png"
        images_list[i] = image_name
        repeated_images_list[i*repeat:(i+1)*repeat] = [image_name]*repeat
        image.save(image_name, "PNG")
        i = i + 1

    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, images_list = repeated_images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat,
                reset_each = True,
                )
    # calculate flows
    i = 0
    original_vectors = [None] * len(population)
    fidelity = [None] * len(population)
    for input_image in images_list:
        prediction_image_path = prediction_dir + str(i).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
        fidelity[i] = get_fidelity(input_image, prediction_image_path)
        if results["vectors"]:
            original_vectors[i] = np.asarray(results["vectors"])
        else:
            original_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    # #mirror images
    # mirror_multiple(output_dir + "images/", mirror_images_dir, TransformationType.MirrorAndFlip)
    # #print("mirror images finished")
    # temp_list = sorted(os.listdir(mirror_images_dir))
    # temp_list = temp_list[0:len(images_list)]
    # mirror_images_list = [mirror_images_dir + im for im in temp_list]
    # repeated_mirror_list = [mirror_images_dir + im for im in temp_list for i in range(repeat) ]

    # # predict
    # test_prednet(initmodel = model_name, images_list = repeated_mirror_list, size=size, 
    #             channels = channels, gpu = gpu, output_dir = mirror_dir + "prediction/", skip_save_frames=repeat,
    #             reset_each = True
    #             )
    # # calculate flow
    # i = 0
    # mirrored_vectors = [None] * len(population)
    # for input_image in mirror_images_list:
    #     print(input_image)
    #     prediction_image_path = mirror_dir + "prediction/" + str(i).zfill(10) + ".png"
    #     print(prediction_image_path)
    #     results = lucas_kanade(input_image, prediction_image_path, output_dir+"/mirrored/flow/", save=True)
    #     if results["vectors"]:
    #         mirrored_vectors[i] = np.asarray(results["vectors"])
    #     else:
    #         mirrored_vectors[i] = [[0,0,-1000,0]]
    #     i = i + 1

    # calculate score
    radius_limits = [20,50]
    scores = [None] * len(population)
    for i in range(0, len(population)):
        #score = combined_illusion_score(original_vectors[i], mirrored_vectors[i])
        score = 0
        if(len(original_vectors[i])>0):
            # bonus
            score = score + 0.1
            ratio = plausibility_ratio(original_vectors[i])
            score_0 = ratio[0]
            good_vectors = ratio[1]
            # score = score + score_0
            if(len(good_vectors)>0): 
            #     # bonus
            #     score = score + 0.1
            #     ratio = plausibility_ratio(mirrored_vectors[i])
            #     good_vectors_m = ratio[1]
            #     # print("good_vectors", good_vectors)
            #     score_1 = mirroring_score(good_vectors, good_vectors_m)
            #     if score_1 < 10:
            #         # bonus
            #         score = score + 10 - score_1

                score_2 = circle_tangent_ratio(good_vectors, limits = radius_limits)
                # f = len(good_vectors)
                # if(f>20):
                #     f = 20
                score = score_2[0] + score_2[1] #*score_2*len(good_vectors)
                # score_3 = strength_number(good_vectors)
                # score = score + score_2 + score_3

        scores[i] =[i, score]

    # normalize everything, and reverse the scores that should be minimized
    # for i in range(0, len(scores)):
    #     score = scores[i]
    #     # avoid comparing different species
    #     s0 = -score[1][0]
    #     s1 = score[1][1]
    #     s3 = score[1][2]
    #     total = (s0+s1+s3+ fidelity[i]*3)

    #     scores[i] = [score[0], total]

    print("scores",scores)
    i = 0
    for genome_id, genome in population:
        genome.fitness = scores[i][1]
        i = i+1

# take the flow vectors origins and change the pixels
def neat_illusion(input_image, output_dir, model_name, checkpoint = None):
    repeat = 6
    limit = 1
    w = 160
    h = 120
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0
    c_dim = 3
    scaling = 10

    best_dir = output_dir + "best/"
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    leaf_names = ["x","y","r"]
    out_names = ["r","g","b"]

    x_dat, y_dat, r_dat = create_grid(w, h, scaling)
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())
    inp_r = torch.tensor(r_dat.flatten())

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "chainer_prednet/neat.cfg")

    def eval_genomes(genomes, config):
        get_fitnesses_neat(genomes, model_name, config)

    checkpointer = neat.Checkpointer(50)

    # Create the population, which is the top-level object for a NEAT run.
    if not checkpoint:
        p = neat.Population(config)
    else:
        p = checkpointer.restore_checkpoint(checkpoint)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(checkpointer)

    # Run for up to x generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    delta_w_node = create_cppn(
        winner,
        config,
        leaf_names,
        out_names
    )

    image_array = np.zeros(((w,h,3)))
    c = 0
    for node_func in delta_w_node:
        pixels = node_func(x=inp_x, y=inp_y, r = inp_r)
        pixels_np = pixels.numpy()
        image_array[:,:,c] = np.reshape(pixels_np, (w, h))
        c = c + 1

    img_data = np.array(image_array*255.0, dtype=np.uint8)
    image =  Image.fromarray(np.reshape(img_data,(h,w,c_dim)))
    image.save("best_illusion.png")




parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
parser.add_argument('--checkpoint', '-c', help='path of checkpoint to restore')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

neat_illusion(args.input, output_dir, args.model, args.checkpoint)

