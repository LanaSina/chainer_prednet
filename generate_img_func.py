import numpy as np
import os
import itertools

def generate_imagelist(args):
    split_ratio = np.array([0,1])
    splits = ["train", "test"]

    n_images = args.n_images
    data_dir = args.data_dir
    input_images_dir = "input_images/"
    im_dir = data_dir + "/" + input_images_dir
    print("im_dir", im_dir)
    image_list = sorted(os.listdir(im_dir))

    if n_images==-1:
        n_images = len(image_list)
    else:
    	image_list = image_list[:n_images]

    limits = split_ratio*n_images
     print("im_dir 2 ", im_dir)
    print(n_images, " images")
    print(limits)

    train_list_file = os.path.join(data_dir, "train_list.txt")
    test_list_file = os.path.join(data_dir, "test_list.txt")

    print('\nSave %s' % train_list_file)
    with open(train_list_file, 'w') as f:
        f.write(input_images_dir)
        tmp = "\n" + input_images_dir
        f.write(tmp.join(image_list[:int(limits[0])]))

    print('Save %s' % test_list_file)
    # save with repetitions
    lst = list(itertools.chain.from_iterable(itertools.repeat(x, args.rep) for x in image_list[int(limits[0]):]))
    with open(test_list_file, 'w') as f:
        for x in range(0,args.total_rep):
            f.write(input_images_dir)
            tmp = "\n" + input_images_dir
            f.write(tmp.join(lst))
            f.write("\n")
    print("Done.")

