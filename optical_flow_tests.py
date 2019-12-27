import argparse
import cv2
import os
from PredNet.call_prednet import call_prednet
from generate_img_func import generate_imagelist
import numpy as np
from optical_flow.optical_flow import lucas_kanade
from utilities.mirror_images import mirror

# This file runs a mirrored flow analysis to find which images contain illusions

def run_prednet(input_path, model_name, limit, repeat):
    print("run prednet")
    l = limit*10
    class PrednetArgs:
        images = input_path
        initmodel = model_name
        input_len = l
        ext = 0
        ext_t = -1
        bprop = 20
        save = 10000
        period = 1000000
        test = True
        skip_save_frames = repeat

        sequences = ''
        gpu = 0 # -1
        root = "."
        resume = ''
        size = '160,120'
        channels = '3,48,96,192'
        offset = "0,0"

    # only save last image
    # %run 'chainer_prednet/PredNet/main.py' --images 'imported' --initmodel 'fpsi_500000_20v.model' --input_len 10 --test 
    prednet_args = PrednetArgs()
    call_prednet(prednet_args)

def make_img_list(input_path, limit, repeat):
    print("create image list")
    # 'chainer_prednet/generate_imagelist.py' 'imported/' '1' -tr 10
    parser = argparse.ArgumentParser(description='generate_imagelist args')
    class ImglistArgs:
        data_dir = input_path
        n_images = limit
        rep = repeat
        total_rep = 1

    imagelist_args = ImglistArgs()
    generate_imagelist(imagelist_args)

# return true if there are some strong vectors in there
def strong_vectors(vectors):
    # to be affined
    threshold = 0.02
    # data is rows of [x, y, dx, dy]
    if (sum(abs(vectors[2]))>threshold);
        return True
    if (sum(abs(vectors[3]))>threshold);
        return True  
    return False

def save(results, mirrored_results, filename, output_path):
    name = filename.split("/")
    name = name[len(name)-1]
    temp = name.split(".")

    output_file = output_path + "flow/original/" + temp[0] + ".png"
    print("saving", output_file)
    cv2.imwrite(output_file, results["image"])    
    output_file = output_path + "flow/mirrored/" + temp[0] + ".png"
    cv2.imwrite(output_file, mirrored_results["image"])    
    output_file = output_path + "/csv/" + temp[0] +".csv"
    with open(output_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(results["vectors"])

# returns true if one direction seems to have a motion illusion
def mirror_test(vectors, mirrored_vectors):
    threshold = 0.02
    # sum quarter by quarter
    w = 160
    h = 120
    step = 10 #px
    x = 0
    while x<w :
        y = 0
        subset_cond = (vectors[0] >= x and vectors[0] < x + step)
        subset_x = vectors[subset_cond]
        subset_cond = (mirrored_vectors[0] >= (w-x) and mirrored_vectors[0] < (w-x-step))
        subset_xm = mirrored_vectors[subset_cond]

        while y<h :
            subset_cond = (subset_x[1] >= y and subset_x[1] < y + step)
            subset_y = subset_x[subset_cond]
            subset_cond = (mirrored_vectors[0] >= (h-y) and mirrored_vectors[0] < (h-y-step))
            subset_ym = mirrored_vectors[subset_cond]

            # check x and y separately because of model bias
            dx_mean = np.mean(subset_y[2])
            if dx_mean > threshold :
                vmean = np.mean(subset_y[2]) + np.mean(subset_ym[2])
                if vmean<threshold :
                    return True

            dy_mean = np.mean(subset_y[3])
            if dy_mean > threshold :
                vmean = np.mean(subset_y[3]) + np.mean(subset_ym[3])
                if vmean<threshold :
                    return True
        
            y = y + step

        x = x + step

    return False


def compare_flow(input_image_dir, prediction_image_dir, output_dir, limit):
    # calculate optical flow compared to input
    print("calculate optical flow")
    if not os.path.exists(output_dir+"/flow/original/"):
        os.makedirs(output_dir+"/flow/original/")
    if not os.path.exists(output_dir+"/flow/mirrored/"):
        os.makedirs(output_dir+"/flow/mirrored/")
    if not os.path.exists(output_dir+"/csv"):
        os.makedirs(output_dir+"/csv")

    prediction_image_dir = "result"
    output_image_list = sorted(os.listdir(prediction_image_dir))

    # python optical_flow.py test_20y_0.jpg test_20y_1.jpg -s 0 -l 1 -cc yellow -lc red -s 2 -l 2 -vs 60.0
    for i in range(0,limit):

        # results for original image 
        original_image = input_image_list[i]
        # original input
        original_image_path = os.path.join(input_image_dir, original_image)
        # prediction
        prediction_image_path = prediction_image_dir + "/" + output_image_list[i] 
        results = lucas_kanade(original_image_path, prediction_image_path, output_dir, save=False)

        # reject too small vectors
        if (!strong_vectors(results.vectors)):
            continue
        
        # results for mirrored image 
        original_image = input_image_list[i]
        # original input
        original_image_path = os.path.join(input_image_dir, original_image)
        # prediction
        prediction_image_path = prediction_image_dir + "/" + output_image_list[i] 
        mirrored_results = lucas_kanade(original_image_path, prediction_image_path, output_dir, save=False)
        
        if (!strong_vectors(mirrored_results.vectors)):
            continue

        # analyse the vectors
        if (mirror_test(results["vectors"], mirrored_results["vectors"])):
            # save files and images
            save(results, mirrored_results, original_image)

# process images as static images
def predict_static(input_path, output_dir, model_name, limit, repeat=10):
    input_image_dir = input_path + "/input_images/"
    input_image_list = sorted(os.listdir(input_image_dir))
    if limit==-1:
        limit = len(input_image_list)

    # predict original images
    make_img_list(input_path, limit, repeat)
    run_prednet(input_path, model_name, limit, repeat)
    # predict mirrored images
    # "chainer_prednet/utilities/mirror_images.py" -i "imported/input_images" -o "mirrored"
    mirror_images_path = "mirrored"
    mirror(input_image_dir, mirror_images_path)
    mirror_images_dir = "mirrored/input_images"
    make_img_list(mirror_images_path, limit, repeat)
    run_prednet(mirror_images_path, model_name, limit, repeat)

    # now compare image by image
    compare_flow(input_image_dir, prediction_image_dir, output_dir, limit)


parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='output', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
parser.add_argument('--limit', '-l', type=int, default=-1, help='max number of images')
parser.add_argument('--repeat', '-r', type=int, default=10, help='number of times to repeat image before calculating flow')


args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

predict_static(args.input,output_dir, args.model, args.limit, args.repeat)
