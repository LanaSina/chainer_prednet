import os
from PIL import Image, ImageOps

# mirror all images on vertical and horizontal axes
def mirror(input_path, output_dir, limit=-1):
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
        im_mirror.save(image_name, quality=100)