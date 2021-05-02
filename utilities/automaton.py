import argparse
import cv2
from PIL import Image, ImageOps
import math
import numpy as np
import copy
import imageio

class Cell:
    def __init__(self, input_pixel, x, y, step, image_size):
        self.input_pixel = input_pixel
        self.edge = False
        self.offset = np.zeros((3))
        self.value = input_pixel.copy()
        self.changed = True
        self.contrast_values = np.array([np.min(self.value), np.max(self.value)])
        # cell neighborhood
        self.brightest_neighbor = input_pixel.copy()
        #self.darkest_neighbor = input_pixel.copy()


    def init_neighborhood(self, size):
        self.neighborhood = [None]*size


    def set_neighbor(self, index, n):
        self.neighborhood[index] = n
        # consider brightest of each channel as composite white
        for c in range(3):
            if n.input_pixel[c] > self.brightest_neighbor[c]:
                self.brightest_neighbor[c] = n.input_pixel[c]
            # if n.input_pixel[c] < self.darkest_neighbor[c]:
            #     self.darkest_neighbor[c] = n.input_pixel[c]


    def update_edge(self, edge_contrast):

        # find the common biggest difference between the 3 channels
        temp = np.min(self.brightest_neighbor)
        common_difference = self.brightest_neighbor - temp

        for n in self.neighborhood:
            # compute edge and offset color
            if (abs(np.mean(self.input_pixel) - np.mean(n.input_pixel))) > edge_contrast:
                self.edge = True
                self.offset = (np.max(common_difference) - common_difference)
                self.value = self.input_pixel + self.offset
                self.contrast_values = self.find_contrast()
                

    def find_contrast(self):
        col_max = self.contrast_values[0]
        col_min = self.contrast_values[1]

        for n in self.neighborhood:
            col_max = max(np.max(n.value), col_max)
            col_min = min(np.min(n.value), col_min)

        return [col_min, col_max]


    def  apply_contrast(self):
        col_min = self.contrast_values[0]
        col_max = self.contrast_values[1]

        if(col_min != col_max):
            factor = 255.0/(col_max-col_min)
            new_pixel = (self.value-col_min)*factor
            new_value = color_clip(new_pixel.astype(int))
            self.value = new_value
        else:
            self.value = color_clip(self.value)


    def update(self):
        # reset
        updated = False

        if not self.edge:
            mean_offset = np.zeros((3))
            mean_contrast = np.zeros((2))
            for n in self.neighborhood:
                # find the mean offset
                mean_offset = mean_offset + n.offset
                # mean contrast values
                mean_contrast =  mean_contrast + n.contrast_values

            mean_offset = mean_offset/len(self.neighborhood)
            mean_contrast = mean_contrast/len(self.neighborhood)

            if abs(mean_offset-self.offset).any() > 0.9:
                self.offset = mean_offset
                clipped_value = color_clip(self.input_pixel + self.offset)
                self.value = clipped_value
                updated = True

            if abs(mean_contrast-self.contrast_values).any() > 0.9:
                self.contrast_values = mean_contrast
                self.apply_contrast()
                updated = True
        #else:
            # apply contrast?


        return updated


def image_from_automaton(a):
    size = [len(a), len(a[0])]
    image = np.zeros((size[0], size[1], 3))
    for x in range(size[0]):
        for y in range(size[1]):
            image[x,y] = a[x][y].value

    return image.astype(np.uint8)

def filter(input_path):
    average_image  = Image.open(input_path)#.convert("L")
    average_image = np.array(average_image)
    dim = len(average_image.shape)
    size = (average_image.shape[1],average_image.shape[0])

    step = 2
    edge_contrast = 50
   

    automaton = [[None for _ in range(size[1])] for _ in range(size[0])]
    # init automaton
    for x in range(size[1]):
            for y in range(size[0]):
                # print(x,y,size[1],size[0])
                automaton[x][y] = Cell(average_image[x,y], x, y, step, average_image.shape)

    output_image = image_from_automaton(automaton)
    to_save = Image.fromarray(output_image.astype(np.uint8))
    display(to_save)

    for x in range(size[1]):
        for y in range(size[0]):

            x0 = max(0, x - step)
            y0 = max(0, y - step)
            # subsetting leaves last number out
            x1 = min(size[1], x + step + 1) 
            y1 = min(size[0], y + step + 1)
            automaton[x][y].init_neighborhood((x1-x0)*(y1-y0)-1)

            i = 0
            for cx in range(x0,x1):
                for cy in range(y0,y1):
                    if cx!=x or cy!=y:
                        automaton[x][y].set_neighbor(i, automaton[cx][cy])
                        i = i+1

            automaton[x][y].update_edge(edge_contrast)


    output_image = image_from_automaton(automaton)
    to_save = Image.fromarray(output_image.astype(np.uint8))
    out_path = "initial.png"
    to_save.save(out_path)
    display(to_save)

    updated = True
    frame_id = 0
    video_length = 100
    frames = np.zeros((video_length,size[1],size[0],3)).astype(np.uint8)
    while updated:
        updated = False
        # update automaton
        for x in range(size[1]):
            for y in range(size[0]):
                cell = automaton[x][y]

                if cell.changed:
                    cell.changed = cell.update()
                    if cell.changed:
                        updated = True

        output_image = image_from_automaton(automaton)
        to_show = Image.fromarray(output_image.astype(np.uint8))
        display(to_show)

        if(frame_id<video_length):        
            frames[frame_id] = output_image.astype(np.uint8)
            frame_id = frame_id + 1
            if(frame_id==video_length-1):
                # out = cv2.VideoWriter('./output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
                # for frame in frames:
                #     out.write(frame)
                # out.release()

                imageio.mimwrite('output_filename.mp4', frames , fps = 30)

                print("video saved")

    #     contrasted_image = np.zeros((size[1], size[0], 3))
    #     for x in range(size[1]):
    #         for y in range(size[0]):
    #             # cell neighborhood
    #             x0 = max(0, x - step)
    #             y0 = max(0, y - step)
    #             x1 = min(size[1], x + step + 1) #subsetting leaves last number out
    #             y1 = min(size[0], y + step + 1)

    #             # apply contrast
    #             # can use average_image because previous computation did not change the max
    #             brightest_pixel = [np.max(output_image[x0:x1, y0:y1, 0]), np.max(output_image[x0:x1, y0:y1, 1]), np.max(output_image[x0:x1, y0:y1, 2])]
    #             col_max = np.max(brightest_pixel)
    #             # not sure if should use this
    #             col_min = np.min(output_image[x0:x1, y0:y1]) 

    #             # offset the color
    #             if(col_min != col_max):
    #                 factor = 255.0/(col_max-col_min)
    #                 new_pixel = (output_image[x,y]-col_min)*factor
    #                 contrasted_image[x,y] = color_clip(new_pixel.astype(int))
    #             else:
    #                 contrasted_image[x,y] = output_image[x,y]
        
    #     av = Image.fromarray(contrasted_image.astype(np.uint8)) 
    #     display(av)

    # # save
    # out_path = "filtered.png"
    # av.save(out_path)

def color_clip(pixel):
    pixel = [min(255, pixel[0]), min(255, pixel[1]), min(255, pixel[2])]
    pixel = [max(0, pixel[0]), max(0, pixel[1]), max(0, pixel[2])]
    return pixel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate illusions')
    parser.add_argument('--input', '-i', default='', help='input image')

    args = parser.parse_args()

    filter(args.input)
