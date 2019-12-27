import argparse
import os
from PredNet.call_prednet import call_prednet
from generate_img_func import generate_imagelist
from optical_flow import lucas_kanade


# process images as static images
def predict_static(input_path, output_dir, model_name, limit):
    # how many times to repeat each image
    repeat = 10

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

    # calculate optical flow compared to input
    print("calculate optical flow")
    input_image_dir = input_path + "/input_images/"
    input_image_list = sorted(os.listdir(input_image_dir))
    output_image_list = sorted(os.listdir("results"))

    if limit==-1:
        limit = len(input_image_list)

    # python optical_flow.py test_20y_0.jpg test_20y_1.jpg -s 0 -l 1 -cc yellow -lc red -s 2 -l 2 -vs 60.0
    lucas_kanade()
    for i in range(0,limit):
        original_image = input_image_list[i]
        # original input
        original_image_path = os.path.join(image_dir, original_image)
        # prediction 
        lucas_kanade(image_path, output_image_list[i], output_dir)

      # h, w, color
      current_image = numpy.array(Image.open(image_path).convert('RGB'))


parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='output', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
parser.add_argument('--limit', '-l', type=int, default=-1, help='max number of images')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

predict_static(args.input,output_dir, args.model, args.limit)
