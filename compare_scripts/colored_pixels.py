import argparse
import csv
import numpy as np
import os
from PIL import Image

# save the position and values of >0 pixels
def record(input_path, output_dir, limit):
    write_file = output_dir + "/pixels.csv"
    print("Writing in", write_file)
    w = 160
    h = 120

    input_list = sorted(os.listdir(input_path))
    if limit==0:
        n = len(input_list)
    else:
        n = limit

    with open(write_file, mode='w') as csv_file:
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

# record the previous value of the pixels
# also record the pixel in a 3px radius
def record_previous(input_path, output_dir):
    input_list = sorted(os.listdir(input_path))

    read_file = output_dir + "/pixels.csv"
    write_file = output_dir + "/previous_pixels.csv"
    with open(read_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None) # skip header
        with open(write_file, mode='w') as csv_file:
            fieldnames = ["image","x", "y", "r", "g", "b"]
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for row in reader:
                i = int(row[0])
                if(i==0):
                    continue

                # get pixel from coordinates
                input_image_path = input_path + "/" + input_list[i-1]
                input_image = np.array(Image.open(input_image_path).convert('RGB'))
                pixel = input_image[int(row[2]), int(row[1])]
                # write in file
                row2 = row[0:3]
                row2.extend(pixel)
                writer.writerow(row2)


parser = argparse.ArgumentParser(description='image_copy')
parser.add_argument('--input', '-i', default='', help='Path to input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--limit', '-l', type=int, default=0, help='max number of images')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

#record(args.input, output_dir, args.limit)
record_previous(args.input, output_dir)

