import argparse
import cv2
import csv
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

    images_list = [input_image]*repeat
    print(images_list)

    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, images_list = images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = "result", skip_save_frames=repeat)


 # image = cv2.imread(input_image)

 #    # predict mirrored images
 #    # "chainer_prednet/utilities/mirror_images.py" -i "imported/input_images" -o "mirrored"
 #    mirror_images_path = "mirrored"
 #    mirror_images_dir = "mirrored/input_images"
 #    if not os.path.exists(mirror_images_dir):
 #        os.makedirs(mirror_images_dir)
 #    mirror(input_image_dir, mirror_images_dir, limit, mtype)
 #    make_img_list(mirror_images_path, limit, repeat)
 #    run_prednet(mirror_images_path, model_name, limit, repeat, "mirrored_result")

 #    # mirror it




 #    results = lucas_kanade(original_image_path, prediction_image_path, output_dir+"/original/", save=stype)
 #    vectors = np.asarray(results["vectors"])
 #    flow_score =  calculate_flow_score(vectors)       

 #    #change pixels


    

parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

generate(args.input, output_dir, args.model)
