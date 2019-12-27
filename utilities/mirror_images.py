import argparse
from PIL import Image, ImageOps
import os

# mirror all images on vertical and horizontal axes
def mirror(input_path, output_dir, limit):
    input_image_list = sorted(os.listdir(input_path))
    if limit==-1:
        limit = len(input_image_list)

    for i in range(0,limit):
        current_image = Image.open(input_path+"/"+input_image_list[i]).convert('RGB')
        im_flip = ImageOps.flip(current_image)
        im_mirror = ImageOps.mirror(im_flip)
        image_name = output_dir + "/" + input_image_list[i]
        print("saving", image_name)
        # !!! jpg
        im_flip.save(image_name, quality=100)


parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--output_dir', '-o', default='mirrored', help='Images will be saved in output_dir/input_images')
parser.add_argument('--limit', '-l', type=int, default=-1, help='max number of images')

args = parser.parse_args()
output_dir = args.output_dir+"/input_images" 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

mirror(args.input, output_dir, args.limit)