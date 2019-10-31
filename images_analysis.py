import argparse
import csv
import cv2
import numpy
import os
from PIL import Image
import random


# red[top, bottom] blue[top, bottom]
def top_bottom_diff(current_image, middle):
  # sum red and blue on the 2 halves
  red = [0,0]
  blue = [0,0]
  for i in range(0, current_image.shape[0]):
    for j in range(0, current_image.shape[1]):
      if i<middle: # top
        red[0] = red[0] + current_image[i, j, 0]
        blue[0] = blue[0] + current_image[i, j, 2]
      else:
        red[1] = red[1] + current_image[i, j, 0]
        blue[1] = blue[1] + current_image[i, j, 2]

  return(red, blue)

def motion_half(current_image, middle):
  # sum red and blue on the 2 halves
  red = [0,0]
  blue = [0,0]
  for i in range(0, current_image.shape[0]):
    for j in range(0, current_image.shape[1]):
      if j<middle: # left
        red[0] = red[0] + current_image[i, j, 0]
        blue[0] = blue[0] + current_image[i, j, 2]
      else:
        red[1] = red[1] + current_image[i, j, 0]
        blue[1] = blue[1] + current_image[i, j, 2]


  red[0] = red[0] / (current_image.shape[0]*current_image.shape[1])
  red[1] = red[1] / (current_image.shape[0]*current_image.shape[1])
  blue[0] = blue[0] / (current_image.shape[0]*current_image.shape[1])
  blue[1] = blue[1] / (current_image.shape[0]*current_image.shape[1])

  return(red, blue)

# [red, blue]
def red_blue_average(current_image):
  # average red and blue vlaues
  result = [0,0]
  for i in range(0, current_image.shape[0]):
    for j in range(0, current_image.shape[1]):
      result[0] = result[0] + current_image[i, j, 0]
      result[1] = result[1] + current_image[i, j, 2]
  
  result[0] = result[0] / (current_image.shape[0]*current_image.shape[1])
  result[1] = result[1] / (current_image.shape[0]*current_image.shape[1])

  return(result)

#average by column
def red_blue_green(current_image, output_file):
  print(current_image.shape)
  fieldnames = ['red','blue','green']
  with open(output_dir+output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)

    result = [0,0,0]
    for j in xrange(0, current_image.shape[1]):
      for i in xrange(0, current_image.shape[0]):
        result[0] = result[0] + current_image[i, j, 0]
        result[1] = result[1] + current_image[i, j, 1]
        result[2] = result[2] + current_image[i, j, 2]
      result[0] = result[0] / (current_image.shape[0])
      result[1] = result[1] / (current_image.shape[0])
      result[2] = result[2] / (current_image.shape[0])
      writer.writerow(result)
  print("wrote in ", output_dir)


def red_blue_diff(current_image, next_image):
  # create image with only the increases in blue or red
  # new_image = numpy.zeros(current_image.shape)
  # white background
  new_image = numpy.ones(current_image.shape)
  new_image = new_image*255
  #new_image = current_image;

  for i in xrange(0,current_image.shape[0]):
    for j in xrange(0,current_image.shape[1]):

      rdiff = 1.0*next_image[i, j, 0] - current_image[i, j, 0]
      # gdiff = 1.0*next_image[i, j, 1] - current_image[i, j, 1]
      bdiff = 1.0*next_image[i, j, 2] - current_image[i, j, 2]

      lim = 0
      # if (rdiff>bdiff):
      # if (bdiff<=lim and rdiff>lim):
      if (rdiff>0 and rdiff<100):
        new_image[i, j, 0] = rdiff
        new_image[i, j, 1] = 0
        new_image[i, j, 2] = 0
      # if (bdiff>rdiff):
      # if (rdiff<lim and bdiff>lim):
      # if (bdiff>0 and bdiff<100):
      #   new_image[i, j, 0] = 0
      #   new_image[i, j, 1] = 0
      #   new_image[i, j, 2] = bdiff
      # if (gdiff>bdiff):
      #   new_image[i, j, 1] = gdiff

  return new_image

# count the ratio of blue and red adter bblack-white alternation in a pixel
def sample_average_color(image_dir, output_dir):
  image_list = sorted(os.listdir(image_dir))
  image_count = len(image_list)

  # sample from this square
  w = 0
  h = 90/2
  size = 20

  #better to open several files... but how
  output_file = "bike_4h_left_sample.csv"
  fieldnames = ['r','g','b']

  # output_file = "fpsi_transitions.csv"
  # fieldnames = ['black_tr', 'white_tr']
  with open(output_dir+output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)

    for image_file in image_list[:image_count]:
      image_path = os.path.join(image_dir, image_file)
      
      # h, w, color
      current_image = numpy.array(Image.open(image_path).convert('RGB'))
      r = 0.0
      g = 0.0
      b = 0.0
      for i in range(h,h+size):
        for j in range(w,w+size):
          r = r + current_image[i, j, 0]
          g = g + current_image[i, j, 1]
          b = b + current_image[i, j, 2]

      r = r/(size*size)
      g = g/(size*size)
      b = b/(size*size)
      row = [r,g,b]
      writer.writerow(row)

  print("done")

# mean rgb in the specified area, minus i,j coordinate
def rgb_surround(i,j, size, image):
  rgb = numpy.zeros(3)
  max_i = image.shape[0]
  max_j = image.shape[1]

  i_start = i - size;
  if i_start <0:
    i_start = 0
  j_start = j -size
  if j_start <0:
    j_start = 0

  i_end = i+size
  if i_end >max_i:
    i_end = max_i
  j_end = j+size
  if j_end >max_j:
    j_end = max_j

  for ii in range(i_start,i_end):
    for jj in range(j_start,j_end):
      if (ii!=i) and (jj!=j):
        rgb = rgb + image[ii,jj]

  rgb = rgb / ( (i_end-i_start)*(j_end-j_start) - 1)
  return rgb

# count the ratio of blue and red after black-white alternation in a pixel
def black_white_next(image_dir, output_dir, n):
  image_list = sorted(os.listdir(image_dir))
  image_count =  n

  if(n == -1):
    image_count =  len(image_list)

  # choose some random pixels
  w = 160
  h = 90
  pixels_count = 100
  coordinates = numpy.zeros((pixels_count,2))
  coordinates[:,0] = numpy.random.randint(h, size=pixels_count) 
  coordinates[:,1] = numpy.random.randint(w, size=pixels_count) 

  t = 2 + 1
  previous_images = []
  for i in range(0,t):
    previous_images.append(numpy.zeros((h,w,3)))

  black = numpy.zeros((pixels_count,t))
  white = numpy.zeros((pixels_count,t))
  col_mean = numpy.zeros((pixels_count,t))

  b_t = 55
  w_t = 200
  blue_t = 0
  red_t = 0
  variation = 0.0

  #better to open several files... but how
  output_file = "flickers.csv"
  fieldnames = ['bw_r','bw_g','bw_b','wb_r','wb_g','wb_b', \
  'bw_diff_rt1', 'bw_diff_gt1', 'bw_diff_bt1', 'wb_diff_rt1', 'wb_diff_gt1', 'wb_diff_bt1']

  started = False
  with open(output_dir+output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)

    for image_file in image_list[:image_count]:
      image_path = os.path.join(image_dir, image_file)
      print("read ", image_path)

      # h, w, color
      current_image = numpy.array(Image.open(image_path).convert('RGB'))

      # current_image = cv2.imread(image_path)
      # current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) 

      # w, h ...
      # new_image = numpy.ones(current_image.shape)*255
      pixel_index = 0
      # (black->white , white->black)
      blue_sum = [0,0]
      red_sum = [0,0]
      green_sum = [0,0]
      bw_sum = 1 # avoid /0
      wb_sum = 1
      # bw, wb
      color_tracks = numpy.zeros((2,3,3))
      # sums
      rgb_s = numpy.zeros((2,3))
      # differences in rbg, bw / wb
      diff_rgb = numpy.zeros((2,3))

      for time_index in range(0,t-1):
        previous_images[time_index] = previous_images[time_index+1]
      previous_images[t-1] = current_image

      for pixel in coordinates:
      
        i = int(pixel[0])
        j = int(pixel[1])

        # !!!!! avoid weird typing
        r = 1.0*current_image[i, j, 0]
        g = 1.0*current_image[i, j, 1]
        b = 1.0*current_image[i, j, 2]
      
        # shift measures in time
        for time_index in range(0,t-1):
          black[pixel_index][time_index] = black[pixel_index][time_index+1]
          white[pixel_index][time_index] = white[pixel_index][time_index+1]
          col_mean[pixel_index][time_index] = col_mean[pixel_index][time_index+1]
          
        #reset
        white[pixel_index][t-1] = 0
        black[pixel_index][t-1] = 0

        col_mean[pixel_index][t-1] = (r + g + b)/3.0

        # is it dark
        if (col_mean[pixel_index][t-1] > w_t):
            white[pixel_index][t-1] = 1
        else:
          if ( col_mean[pixel_index][t-1] < b_t ):
            black[pixel_index][t-1] = 1

        global_rgb_s = rgb_surround(i,j,2,current_image) #previous_images[t-2]
        surround_rgb_mean = sum(global_rgb_s)/3.0
        current_rgb_mean = sum(current_image[i,j])/3.0 #previous_images[t-2]
        rgb_difference = abs(surround_rgb_mean-current_rgb_mean)
        diff_threshold = 30

        #ignore pixels that are not contrasted compared to bg
        if(rgb_difference<diff_threshold):
          continue

        # check black/white transitions
        if (black[pixel_index][0] == 1 and white[pixel_index][1] == 1):
          bw_sum = bw_sum + 1
          red_sum[0] = red_sum[0] + r
          blue_sum[0] = blue_sum[0] + b
          green_sum[0] = green_sum[0] + g
          diff_rgb[0] = 1.0*previous_images[1][i,j] - current_image[i,j]

        # white black
        if (white[pixel_index][0] == 1 and black[pixel_index][1] == 1):
          wb_sum = wb_sum + 1
          red_sum[1] = red_sum[1] + r
          blue_sum[1] = blue_sum[1] + b
          green_sum[1] = green_sum[1] + g
          diff_rgb[1] = 1.0*previous_images[1][i,j] - current_image[i,j]

        pixel_index = pixel_index + 1

      # (black->white , white->black)
      # ['bw_r','bw_g','bw_b','wb_r','wb_g','wb_b']
      # 'diff_rt1', 'diff_gt1', 'diff_bt1'
      row = [red_sum[0]/bw_sum, green_sum[0]/bw_sum, blue_sum[0]/bw_sum, \
      red_sum[1]/wb_sum, green_sum[1]/wb_sum, blue_sum[1]/wb_sum]
      row.extend(diff_rgb[0])
      row.extend(diff_rgb[1])
      writer.writerow(row)

      started = True



# count the ratio of blue and red after black-white alternation in a pixel
def random_flicker(image_dir, output_dir, n):
  image_list = sorted(os.listdir(image_dir))
  image_count =  n

  if(n == -1):
    image_count =  len(image_list)

  # choose some random pixels
  w = 160
  h = 90
  pixels_count = 100
  coordinates = numpy.zeros((pixels_count,2))
  coordinates[:,0] = numpy.random.randint(h, size=pixels_count) 
  coordinates[:,1] = numpy.random.randint(w, size=pixels_count) 

  t = 3
  previous_images = []
  for i in range(0,t):
    previous_images.append(numpy.zeros((h,w,3)))

  diff_threshold = 70

  #better to open several files... but how
  output_file = "random_flickers.csv"
  fieldnames = ['r_0','g_0','b_0','r_1','g_1','b_1', \
  'r_2','g_2','b_2', 'r_diff_t02', 'g_diff_t02', 'b_diff_t02',\
  'r_diff_t12', 'g_diff_t12', 'b_diff_t12']

  with open(output_dir+output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)

    for image_file in image_list[2:image_count]:
      image_path = os.path.join(image_dir, image_file)
      print("read ", image_path)

      # h, w, color
      current_image = numpy.array(Image.open(image_path).convert('RGB'))

      pixel_index = 0

      for time_index in range(0,t-1):
        previous_images[time_index] = previous_images[time_index+1]
      previous_images[t-1] = current_image

      for pixel in coordinates:  
        i = int(pixel[0])
        j = int(pixel[1])

        # !!!!! avoid weird typing
        r = 1.0*current_image[i, j, 0]
        g = 1.0*current_image[i, j, 1]
        b = 1.0*current_image[i, j, 2]
      
    
        col_mean  = numpy.mean(current_image[i, j])

        # has it flickered?
        flicker_value = numpy.mean(previous_images[0][i,j]) - numpy.mean(previous_images[1][i,j])
        if (abs(flicker_value) < diff_threshold):
          continue

        # is the current pixel isolated?
        global_rgb_s = rgb_surround(i,j,2,current_image) 
        surround_rgb_mean = sum(global_rgb_s)/3.0
        current_rgb_mean = sum(current_image[i,j])/3.0
        rgb_difference = abs(surround_rgb_mean-current_rgb_mean)

        #ignore pixels that are not contrasted compared to bg
        if(rgb_difference<diff_threshold):
          continue

        pixel_index = pixel_index + 1

        # r g b  at t=0, t=1, t  = 2
        row = []
        row.extend(previous_images[0][i,j])
        row.extend(previous_images[1][i,j])
        row.extend(current_image[i, j])

        # differences
        rgb_diff = 1.0*current_image[i, j] - previous_images[0][i,j]
        row.extend(rgb_diff)
        rgb_diff = 1.0*current_image[i, j] - previous_images[1][i,j]
        row.extend(rgb_diff)

        writer.writerow(row)



output_dir = "image_analysis/black_white_next/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--image_dir', '-i', default='', help='Path to images folder')
parser.add_argument('--n_images', '-n', default=-1, type=int, help='number of images to process')

args = parser.parse_args()

# # average red and blues by column
# image_path = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/results/" + dataset + "test_20y_0.jpg"
# current_image = numpy.array(Image.open(image_path).convert('RGB'))
# result = red_blue_green(current_image, 'colors_average_20.csv')

# # flickers
random_flicker(args.image_dir, output_dir, args.n_images)
# black_white_next(args.image_dir, output_dir, args.n_images)


# average colors on sample square
#sample_average_color(args.image_dir, output_dir)

