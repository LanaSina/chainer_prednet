import argparse
from PIL import Image, ImageOps
import math
import numpy as np



def filter(input_path):
    average_image  = Image.open(input_path)#.convert("L")
    average_image = np.array(average_image)
    dim = len(average_image.shape)
    size = (average_image.shape[1],average_image.shape[0])

    step = 2
    min_col_var = 1000
    # actually neighborhood size
    cell_size = step*2 + 1
    x_cells = math.ceil(size[0]/cell_size)
    y_cells = math.ceil(size[1]/cell_size)

    if dim==2:
        post_grain_image = np.zeros((size[1], size[0]))
        subtracted = np.zeros((size[1], size[0]))
    else:
        post_grain_image = np.zeros((size[1], size[0], 3))
        subtracted = np.zeros((size[1], size[0], 3))




    # if this pixel is an edge
    # let's just remove common colors and see what happens
    edge_contrast = 50
    c_step = step
    for x in range(size[1]):
        for y in range(size[0]):
            edge = False                              
            pixel = average_image[x,y] 
            post_grain_image[x,y] = pixel

            # cell neighborhood
            x0 = max(0, x - c_step)
            y0 = max(0, y - c_step)
            x1 = min(size[1], x + c_step + 1) #subsetting leaves last number out
            y1 = min(size[0], y + c_step + 1)


            # consider brightest as white
            brightest_pixel = [np.max(average_image[x0:x1, y0:y1, 0]), np.max(average_image[x0:x1, y0:y1, 1]), np.max(average_image[x0:x1, y0:y1, 2])]
            # find the common biggest difference between the 3 channels
            temp = np.min(brightest_pixel)
            common_difference = brightest_pixel - temp
            # this can end up being an array
            highest_channel = np.array([-1])
            # common_difference = None
            for cx in range(x0,x1):
                for cy in range(y0,y1):

                    if (abs(np.mean(pixel) - np.mean(average_image[cx,cy]))) > edge_contrast:
                        edge = True

            if edge:
                adding = (np.max(common_difference) - common_difference)
                new_pixel = pixel + adding
                post_grain_image[x,y] = new_pixel
                subtracted[x,y] = adding
                #print(common_difference, pixel, new_pixel)

            if not edge:
                for cx in range(x0,x1):
                    for cy in range(y0,y1):
                        # if anywhere else was an edge and got added colors
                        # add them here too
                        if subtracted[cx,cy].any() != 0:
                            # get the unfiltered color
                            new_pixel = pixel  + subtracted[cx,cy]
                            post_grain_image[x,y] = new_pixel

            # apply contrast
            # uint to float
            greyscale = 1.0*average_image[x0:x1, y0:y1, 0] + average_image[x0:x1, y0:y1, 1] + average_image[x0:x1, y0:y1, 2]
            greyscale = greyscale/3
            col_max = np.max(greyscale)
            col_min = np.min(greyscale)
            # center the color
            if(col_min != col_max):
                factor = 255/col_max
                new_pixel = (average_image[x,y]-col_min)*factor
                #print(col_min, col_max, factor, average_image[x,y], new_pixel)
                post_grain_image[x,y] = color_clip(new_pixel.astype(int))
                #print(new_pixel, post_grain_image[x,y])




     # save
    out_path = "test.png"

    av = Image.fromarray(post_grain_image.astype(np.uint8))
    #av = av.convert("L")

    av.save(out_path)

def color_clip(pixel):
    pixel = [min(255, pixel[0]), min(255, pixel[1]), min(255, pixel[2])]
    pixel = [max(0, pixel[0]), max(0, pixel[1]), max(0, pixel[2])]
    return pixel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate illusions')
    parser.add_argument('--input', '-i', default='', help='input image')

    args = parser.parse_args()

    filter(args.input)
