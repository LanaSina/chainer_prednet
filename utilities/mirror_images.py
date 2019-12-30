import enum
import os
from PIL import Image, ImageOps


class TransformationType(enum.Enum):
   Mirror = 0
   Flip = 2
   MirrorAndFlip = 3


# mirror all images on vertical and horizontal axes
def mirror(input_path, output_dir, limit=-1, mtype = TransformationType.Mirror):
    input_image_list = sorted(os.listdir(input_path))
    if limit==-1:
        limit = len(input_image_list)

    for i in range(0,limit):
        current_image = Image.open(input_path+"/"+input_image_list[i]).convert('RGB')
        if (mtype == TransformationType.Mirror or mtype == TransformationType.MirrorAndFlip):
            current_image = ImageOps.mirror(current_image)
        if (mtype == TransformationType.Flip or mtype == TransformationType.MirrorAndFlip):
            current_image = ImageOps.flip(current_image)
        
        image_name = output_dir + "/" + input_image_list[i]
        print("saving", image_name)
        # !!! jpg
        current_image.save(image_name, quality=100)