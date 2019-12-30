import enum
import os
from PIL import Image, ImageOps


class TransformationType(enum.Enum):
   Mirror = 0
   Flip = 1
   MirrorAndFlip = 2


# mirror all images on vertical and horizontal axes
def mirror(input_path, output_dir, limit=-1, mtype = TransformationType.Mirror):
    transform = TransformationType(mtype)
    input_image_list = sorted(os.listdir(input_path))
    if limit==-1:
        limit = len(input_image_list)

    for i in range(0,limit):
        current_image = Image.open(input_path+"/"+input_image_list[i]).convert('RGB')
        if (transform == TransformationType.Mirror or transform == TransformationType.MirrorAndFlip):
            current_image = ImageOps.mirror(current_image)
        if (transform == TransformationType.Flip or transform == TransformationType.MirrorAndFlip):
            current_image = ImageOps.flip(current_image)
        
        image_name = output_dir + "/" + input_image_list[i]
        print("saving", image_name)
        # !!! jpg
        current_image.save(image_name, quality=100)