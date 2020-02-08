import argparse
import cv2
import csv
import os

# take the flow vectors origins and change the pixels
def destroy_flow(input_image, output_dir):
    #read image
    image = cv2.imread(input_image)
    
    #read csv
    dot_split = input_image.split(".")
    slash_split = dot_split[0].split("/")
    csv_name =  '/'.join(slash_split[0:len(slash_split)-1])
    csv_name = csv_name + "/csv/" + slash_split[len(slash_split)-1] + ".csv"
    print("csv_name",csv_name)

    #change pixels


    

parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

destroy_flow(args.input,output_dir)
