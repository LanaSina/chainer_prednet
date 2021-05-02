import argparse
from PIL import Image, ImageOps
import math
import numpy as np



def filter(input_path):
    average_image  = Image.open(input_path)#.convert("L")
    average_image = np.array(average_image)
    dim = len(average_image.shape)
    size = (average_image.shape[1],average_image.shape[0])

    step = 5
    min_col_var = 1000
    # actually neighborhood size
    cell_size = step*2 + 1
    x_cells = math.ceil(size[0]/cell_size)
    y_cells = math.ceil(size[1]/cell_size)

    if dim==2:
        post_grain_image = np.zeros((size[1], size[0]))
    else:
        post_grain_image = np.zeros((size[1], size[0], 3))

    # let's just remove common colors and see what happens
    c_step = step
    for x in range(size[1]):
        for y in range(size[0]):
            pixel = average_image[x,y]

            # cell neighborhood
            x0 = max(0, x - c_step)
            y0 = max(0, y - c_step)
            x1 = min(size[1], x + c_step + 1) #subsetting leaves last number out
            y1 = min(size[0], y + c_step + 1)

            col_var = np.var(average_image[x0:x1, y0:y1])
            if(col_var>min_col_var):
                # # the maximal common color
                if dim == 3:
                    col = [np.min(average_image[x0:x1, y0:y1, 0]), np.min(average_image[x0:x1, y0:y1, 1]), np.min(average_image[x0:x1, y0:y1, 2])]
                else:
                    col = np.min(average_image[x0:x1, y0:y1])

                new_pixel = pixel - col
                post_grain_image[x,y] = new_pixel

                # # or add what is missing
                # col = [np.max(average_image[x0:x1, y0:y1, 0]), np.max(average_image[x0:x1, y0:y1, 1]), np.min(average_image[x0:x1, y0:y1, 2])]
                # col = np.ones((dim)) * 255 - col
                # new_pixel = pixel + col
                # post_grain_image[x,y] = new_pixel

            else:
                post_grain_image[x,y] = pixel


    # c_step = step
    # for x in range(size[1]):
    #     for y in range(size[0]):
    #         pixel = average_image[x,y]
    #         # cell neighborhood
    #         x0 = max(0, x - c_step)
    #         y0 = max(0, y - c_step)
    #         x1 = min(size[1], x + c_step + 1) #subsetting leaves last number out
    #         y1 = min(size[0], y + c_step + 1)


    #         # make color a function of the variance at that neighbrhood
    #         col_var = np.var(average_image[x0:x1, y0:y1])
    #         col_mean = np.mean(average_image[x0:x1, y0:y1])
    #         # difference from the mean
    #         d = pixel - col_mean
    #         # max = c2/4  (multiplied by (𝑁−1)/𝑁
    #         n = (step+1)*(step+1)
    #         max_var = 255*255/4.0 * ( n-1 )/n
    #         # print(col_var, max_var)

    #         #print(col_var,col_mean, max_var)
    #         # high variance = lots of contrast => few bins => exaggerated extremes
    #         # low variance => many bins  = spread out values
    #         # max_bins = 255/2 around the mean?
    #         bins = math.ceil(col_var*127/max_var)
    #         # print("bins",bins)

    #         # variance
    #         if bins > 0:
    #             # print("or px", pixel)
    #             # which bin is this pixel in
    #             bin_n = math.floor(pixel*bins/255)
    #             # print("bins", bins, "bin n", bin_n)
    #             if pixel > col_mean:
    #                 pixel = col_mean + bin_n*127/bins
    #                 pixel = min(pixel,255)
    #             else :
    #                 pixel = col_mean - bin_n*127/bins
    #                 pixel = max(pixel,0)
    #             # print("after px",pixel)
                

    #         pixel = int(pixel)
    #         post_grain_image[x,y] = pixel


     # save
    out_path = "test.png"

    av = Image.fromarray(post_grain_image.astype(np.uint8))
    #av = av.convert("L")

    av.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate illusions')
    parser.add_argument('--input', '-i', default='', help='input image')

    args = parser.parse_args()

    filter(args.input)
