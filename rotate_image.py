import os
from PIL import Image

# Create an Image object from an Image
input_image  = Image.open("./clean_disk.png")
output_directory = "./benham_frames/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Rotate it by x degrees
rotation = 20
#1 for clockwise, -1 for counter-clockwise
rotation_direction = 1

n = 360//rotation
#rotated = input_image #.convert('RGB')
non_transparent = Image.new('RGB',input_image.size,(255,255,255))
non_transparent.paste(input_image,(0,0), mask = input_image)
non_transparent.save("./benham_frames/0.jpg")

for i in range(1,n):
	rotated = input_image.rotate(rotation_direction*i*rotation)
	non_transparent = Image.new('RGB',input_image.size,(255,255,255))
	non_transparent.paste(rotated,(0,0), mask = rotated)
	non_transparent.save("./benham_frames/"+  str(i) + ".jpg")