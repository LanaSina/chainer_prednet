import argparse
import csv
import numpy as np
import os
from PIL import Image

# save the position and values of >0 pixels
def record(input_path, output_dir, limit):
    csv_file = output_dir + "/pixels.csv"
    print("Writing in", csv_file)
    w = 160
    h = 120

    input_list = sorted(os.listdir(input_path))
    if limit==0:
        n = len(input_list)
    else:
        n = limit


    with open(csv_file, mode='w') as csv_file:
        fieldnames = ["image","x", "y", "r", "g", "b"]
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        for i in range(0,n):
            input_image_path = input_path + "/" + input_list[i]
            input_image = np.array(Image.open(input_image_path).convert('RGB'))

            for x in range(0,w):
                for y in range(0,h):
                    if sum(input_image[y,x])>0:
                        row = [str(i),str(x),str(y)]
                        row.extend(input_image[y,x])
                        writer.writerow(row)


parser = argparse.ArgumentParser(description='image_copy')
parser.add_argument('--input', '-i', default='', help='Path to input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--limit', '-l', type=int, default=0, help='max number of images')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

record(args.input, output_dir, args.limit)
