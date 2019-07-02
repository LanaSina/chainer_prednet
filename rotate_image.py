import os
from PIL import Image

# Create an Image object from an Image
input_image  = Image.open("./benham.jpg")
output_directory = "./benham_frames/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Rotate it by x degrees
rotation = 20
#1 for clockwise, -1 for counter-clockwise
rotation_direction = 1

n = 360//rotation
for i in range(1,n):
	rotated = input_image.rotate(rotation_direction*rotation)
	rotated.save("./benham_frames/"+  str(i) + ".jpg")