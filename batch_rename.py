import argparse
import os

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('image_dir', action='store', nargs='?', help='Path to prednet output images')
args = parser.parse_args()

path = args.image_dir
for filename in os.listdir(path):
    num = filename.split('_')[1].split('y')[0]
    print(num)
    num = num.zfill(4)
    new_filename = num + ".png"
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))