import numpy as np, random
from PIL import Image


def setup(w, h):
    
    def randColor():
        return np.array([random.random(), random.random(), random.random()]).reshape((1, 1, 3))

    def safeDivide(a, b):
        return np.divide(a, np.maximum(b, 0.001))


    def getX(): 
        return np.linspace(0.0, 1.0, w).reshape((1, w, 1))

    def getY():
        return np.linspace(0.0, 1.0, h).reshape((h, 1, 1))

    # ends up like func(func(func(getX or getY)))
    def buildImg(functions, depth = 0):
        depthMin = 2
        depthMax = 10
        funcs = [f for f in functions if
                    (f[0] > 0 and depth < depthMax) or
                    (f[0] == 0 and depth >= depthMin)]
        nArgs, func = random.choice(funcs)
        args = [buildImg(functions, depth + 1) for n in range(nArgs)]
        return func(*args)


    functions = [(0, randColor),
                 (0, getX),
                 (0, getY),
                 (1, np.sin),
                 (1, np.cos),
                 (2, np.add),
                 (2, np.subtract),
                 (2, np.multiply),
                 (2, safeDivide)]
    img = buildImg(functions)
    return img

def get_random_image(w, h):
    img = setup(w,h)
    
    # Ensure it has the right dimensions, dX by dY by 3
    img = np.tile(img, (int(h / img.shape[0]), int(w / img.shape[1]), int(3 / img.shape[2])))

    # Convert to 8-bit, send to PIL and save
    img8Bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    return Image.fromarray(img8Bit)

def get_random_image_array(w, h):
    img = setup(w,h)
    
    # Ensure it has the right dimensions, dX by dY by 3
    img = np.tile(img, (int(h / img.shape[0]), int(w / img.shape[1]), int(3 / img.shape[2])))

    # Convert to 8-bit, send to PIL and save
    img8Bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    return img8Bit


w = 160
h = 120
image = get_random_image(w,h)
image.save('output.bmp')