import enum
import os
from PIL import Image, ImageOps


class TransformationType(enum.Enum):
   Mirror = 0
   Flip = 1
   MirrorAndFlip = 2


# mirror all images on vertical and horizontal axes
def mirror(input_image, output_dir, lossless, mtype = TransformationType.Mirror):
    transform = TransformationType(mtype)

    current_image = Image.open(input_image).convert('RGB')
    if (transform == TransformationType.Mirror or transform == TransformationType.MirrorAndFlip):
        current_image = ImageOps.mirror(current_image)
    if (transform == TransformationType.Flip or transform == TransformationType.MirrorAndFlip):
        current_image = ImageOps.flip(current_image)
    
    if lossless:
        name = input_image.split("/")
        name = name[len(name)-1]
        temp = name.split(".")
        image_name = output_dir + "/" + temp[0] + ".png"
        print("saving", image_name)
        current_image.save(image_name, "PNG")
    else:
        image_name = output_dir + "/" + input_image
        print("saving", image_name)
        current_image.save(image_name, quality=100)


def mirror_multiple(input_path, output_dir, mtype = TransformationType.Mirror):
    transform = TransformationType(mtype)
    input_image_list = sorted(os.listdir(input_path))
    if limit==-1:
        limit = len(input_image_list)

    for i in range(0,limit):
        input_image = input_path+"/"+input_image_list[i]
        mirror(input_image, output_dir, False, limit, mtype)