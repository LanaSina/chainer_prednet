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

    # 5/10 = 0.5 
    # 10 * 

def combined_illusion_score(vectors, m_vectors):
    # check vector alignements
    sum_v = [0,0]
    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15 or norm==0: 
            continue
        sum_v = [sum_v[0] + vector[2], sum_v[1] + vector[3]]

    sum_mv = [0,0]
    for vector in m_vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15 or norm==0: 
            continue
        sum_mv = [sum_mv[0] + vector[2], sum_mv[1] + vector[3]]

    s0x = sum_v[0] + sum_mv[0]
    s0y = sum_v[1] + sum_mv[1]
    s1 = abs(sum_v[0]) +  abs(sum_v[1]) +  abs(sum_mv[0]) +  abs(sum_mv[1])
    s2 = len(vectors) + len(m_vectors)

    return [s0x + s0y, s1, s2]

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


# w h 
def initial_population(img_shape, n_individuals=2):
     init_population = [None]*n_individuals

     for indv_num in range(n_individuals):
         # Randomly generating initial population chromosomes genes values.
         init_population[indv_num] = generate_random_image(img_shape[1], img_shape[0])
         # #get_random_image_array(img_shape[0], img_shape[1])

     return init_population

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


def get_best(population, n, model_name, limit, id=0):
    print("get best")
    output_dir = "temp" + str(id) + "/"
    repeat = 10
    size = [160,120]
    channels = [3,48,96,192]
    gpu = 0
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
    i = 0
    for image_array in population:
        image = Image.fromarray(image_array)
        image_name = output_dir + "images/" + str(i).zfill(10) + ".png"
        images_list[i] = image_name
        repeated_images_list[i*repeat:(i+1)*repeat] = [image_name]*repeat
        image.save(image_name, "PNG")
        i = i + 1

    #print("saved temporarily", images_list)
    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, images_list = repeated_images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat,
                reset_each = True,
                )
    # calculate flows
    i = 0
    original_vectors = [None] * len(population)
    for input_image in images_list:
        prediction_image_path = prediction_dir + str(i).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
        if results["vectors"]:
            original_vectors[i] = np.asarray(results["vectors"])
        else:
            original_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    #mirror images
    mirror_multiple(output_dir + "images/", mirror_images_dir, TransformationType.MirrorAndFlip)
    #print("mirror images finished")
    temp_list = sorted(os.listdir(mirror_images_dir))
    mirror_images_list = [mirror_images_dir + im for im in temp_list[0:limit]]
    repeated_mirror_list = [mirror_images_dir + im for im in temp_list[0:limit] for i in range(repeat) ]

    # print("mirrored", mirror_images_list)
    # predict
    #print("predict mirror images")
    test_prednet(initmodel = model_name, images_list = repeated_mirror_list, size=size, 
                channels = channels, gpu = gpu, output_dir = mirror_dir + "prediction/", skip_save_frames=repeat,
                reset_each = True
                )
    # calculate flow
    #print("mirror images flow")
    i = 0
    mirrored_vectors = [None] * len(population)
    #print("len(population)", len(population))
    #print("len(ext_mlist)", len(ext_mlist))
    for input_image in mirror_images_list:
        print(input_image)
        prediction_image_path = mirror_dir + "prediction/" + str(i).zfill(10) + ".png"
        print(prediction_image_path)
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/mirrored/flow/", save=True)
        if results["vectors"]:
            mirrored_vectors[i] = np.asarray(results["vectors"])
        else:
            mirrored_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    print("scores")
    # calculate score
    scores = [None] * len(population)
    sums = [1,1,1]
    for i in range(0, len(population)):
        # s0 = illusion_score(original_vectors[i])
        # s1 = illusion_score(mirrored_vectors[i], mirrored=True, flipped=True)
        # scores[i] =[i, score]
        score = combined_illusion_score(original_vectors[i], mirrored_vectors[i])
        sums[0] = sums[0] + score[0]
        sums[1] = sums[1] + score[1]
        sums[2] = sums[2] + score[2]
        scores[i] =[i, score]

    # print(scores)
    #   print(sums)
    # normalize everything, and reverse the scores that should be minimized
    for i in range(0, len(scores)):
        score = scores[i]
        s0 = 1 - (score[1][0]/sums[0])
        s1 = (score[1][1]/sums[1])
        s3 = (score[1][2]/sums[2])
        scores[i] = [score[0], (s0+s1+s3)/3]

    print(scores)

    sorted_scores = sorted(scores, key=lambda x:x[1], reverse = True)
    print("sorted_scores ", sorted_scores)
    results = [population[i[0]] for i in sorted_scores[0:n]]

    return results

# take the flow vectors origins and change the pixels
def generate(input_image, output_dir, model_name):
    repeat = 6
    limit = 1
    size = [160,120]
    channels = [3,48,96,192]
    gpu = 0

    pop_size = 2
    best_dir = output_dir + "best/"
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    next_population = initial_population(size, pop_size)

    for i in range(0,500):
        print("len(next_population)", len(next_population))
        best = get_best(next_population, 2, model_name, limit=len(next_population))
       #print("len(best)",len(best))
        im = 0
        for image_array in best:
            image = Image.fromarray(image_array)
            image.save(best_dir + str(im).zfill(10) +'.png')
            im = im+1
        next_population = crossover(best, n_offspring=2, mutation_ratio=0.5)
        # add 2 new images
        next_population.extend(initial_population(size, 2))        
        # add parents
        next_population.extend(best)

# population:  [id, net]
def get_fitnesses_neat(population, model_name, limit, id=0):
    print("fitnesses of ", len(population))
    output_dir = "temp" + str(id) + "/"
    repeat = 10
    size = [160,120]
    channels = [3,48,96,192]
    gpu = 0

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
    i = 0
    for genome_id, genome in population:
        image_array = np.zeros(((w,h,3)))
        c = 0
        for node_func in delta_w_node:
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
    for input_image in images_list:
        prediction_image_path = prediction_dir + str(i).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
        if results["vectors"]:
            original_vectors[i] = np.asarray(results["vectors"])
        else:
            original_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    #mirror images
    mirror_multiple(output_dir + "images/", mirror_images_dir, TransformationType.MirrorAndFlip)
    #print("mirror images finished")
    temp_list = sorted(os.listdir(mirror_images_dir))
    mirror_images_list = [mirror_images_dir + im for im in temp_list]
    repeated_mirror_list = [mirror_images_dir + im for im in temp_list for i in range(repeat) ]

    # predict
    test_prednet(initmodel = model_name, images_list = repeated_mirror_list, size=size, 
                channels = channels, gpu = gpu, output_dir = mirror_dir + "prediction/", skip_save_frames=repeat,
                reset_each = True
                )
    # calculate flow
    i = 0
    mirrored_vectors = [None] * len(population)
    for input_image in mirror_images_list:
        print(input_image)
        prediction_image_path = mirror_dir + "prediction/" + str(i).zfill(10) + ".png"
        print(prediction_image_path)
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/mirrored/flow/", save=True)
        if results["vectors"]:
            mirrored_vectors[i] = np.asarray(results["vectors"])
        else:
            mirrored_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    print("scores")
    # calculate score
    scores = [None] * len(population)
    sums = [1,1,1]
    for i in range(0, len(population)):
        score = combined_illusion_score(original_vectors[i], mirrored_vectors[i])
        sums[0] = sums[0] + score[0]
        sums[1] = sums[1] + score[1]
        sums[2] = sums[2] + score[2]
        scores[i] =[i, score]

    # normalize everything, and reverse the scores that should be minimized
    for i in range(0, len(scores)):
        score = scores[i]
        s0 = 1 - (score[1][0]/sums[0])
        s1 = (score[1][1]/sums[1])
        s3 = (score[1][2]/sums[2])
        scores[i] = [score[0], (s0+s1+s3)/3]

    print(scores)
    i = 0
    for genome_id, genome in population:
        genome.fitness = score[i][1]
        i = i+1

# take the flow vectors origins and change the pixels
def neat_illusion(input_image, output_dir, model_name):
    repeat = 6
    limit = 1
    w = 160
    h = 120
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0
    c_dim = 3
    scaling = 10

    pop_size = 2
    best_dir = output_dir + "best/"
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    leaf_names = ["x","y","r"]
    out_names = ["r","g","b"]

    x_dat, y_dat, r_dat = create_grid(w, h, scaling)
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())
    inp_r = torch.tensor(r_dat.flatten())
   

    next_population = initial_population(size, pop_size)

    for i in range(0,500):
        print("len(next_population)", len(next_population))
        best = get_best(next_population, 2, model_name, limit=len(next_population))
       #print("len(best)",len(best))
        im = 0
        for image_array in best:
            image = Image.fromarray(image_array)
            image.save(best_dir + str(im).zfill(10) +'.png')
            im = im+1
        next_population = crossover(best, n_offspring=2, mutation_ratio=0.5)
        # add 2 new images
        next_population.extend(initial_population(size, 2))        
        # add parents
        next_population.extend(best)

    def eval_genomes(genomes, config):
        get_fitnesses_neat(genomes, model_name)

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

    image_array = np.zeros(((w,h,3)))
    c = 0
    for node_func in delta_w_node:
        pixels = node_func(x=inp_x, y=inp_y, r = inp_r)
        pixels_np = pixels.numpy()
        image_array[:,:,c] = np.reshape(pixels_np, (w, h))
        c = c + 1

    img_data = np.array(image_array*255.0, dtype=np.uint8)
    image =  Image.fromarray(np.reshape(img_data,(h,w,c_dim)))
    image.save("bet_illusion.png")




parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

neat_illusion(args.input, output_dir, args.model)

