import argparse
import os
from datetime import datetime
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import chainer.computational_graph as c
from tb_chainer import SummaryWriter, NodeName, utils
import net

# what is this doing?
def load_list(path, root):
    # how is that a list of tuples and not just a list of images?
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples

def read_image(full_path, size, offset):
    image = np.asarray(Image.open(full_path)).transpose(2, 0, 1)
    # // is int division
    top = offset[1] + (image.shape[1]  - size[1]) // 2
    left = offset[0] + (image.shape[2]  - size[0]) // 2
    bottom = size[1] + top
    right = size[0] + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image /= 255
    return image

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)

writer = SummaryWriter('runs/test')#+datetime.now().strftime('%B%d  %H:%M:%S'))

def save_model(count):
    print('save the model')
    serializers.save_npz('models/' + str(count) + '.model', model)
    print('save the optimizer')
    serializers.save_npz('models/' + str(count) + '.state', optimizer)
    for name, param in model.predictor.namedparams():
        writer.add_histogram(name, chainer.cuda.to_cpu(param.data), count)
    writer.add_scalar('loss', float(model.loss.data), count)

def train_image_list(imagelist, model, channels, size, gpu, period, save, bprop):
    xp = cuda.cupy if gpu >= 0 else np

    batchSize = 1
    x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
    y_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
    if len(imagelist) == 0:
        print("Not found images.")
        return

    x_batch[0] = read_image(imagelist[0], size, offset)
    for i in range(1, len(imagelist)):
        y_batch[0] = read_image(imagelist[i], size, offset);
        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))

        if (i + 1) % bprop == 0:
            print("count ", count," frameNo ", i)
            model.zerograds()
            loss.backward()
            loss.unchain_backward()
            loss = 0
            optimizer.update()
            if gpu >= 0:model.to_cpu()
            # write_image(x_batch[0].copy(), 'images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'x.png')
            # write_image(model.y.data[0].copy(), 'images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'y.png')
            # write_image(y_batch[0].copy(), 'images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'z.png')
            if gpu >= 0:model.to_gpu()
            print('loss:' + str(float(model.loss.data)))
            logf.write(str(i) + ', ' + str(float(model.loss.data)) + '\n')

        if (count%save) == 0:
            save_model(count)
        x_batch[0] = y_batch[0]
        count += 1
        if (count>=period):
            save_model(count)
            break


def train_image_folders(sequencelist, prednet, imagelist, model, 
                        channels, size, gpu, period, save, bprop, root):
    logf = open('log.txt', 'w')
    count = 0
    seq = 0
    while count < period:
        prednet.reset_state()
        loss = 0
        imagelist = load_list(sequencelist[seq], root)
        train_image_list(imagelist, model, channels, size, gpu, 
                        period, save, bprop)
        seq = (seq + 1)%len(sequencelist)


def test_image_list(prednet, imagelist, model, output_dir, channels, size, offset, gpu, skip_save_frames=0, 
    extension_start=0, extension_duration=100):

    xp = cuda.cupy if gpu >= 0 else np

    prednet.reset_state()
    loss = 0
    batchSize = 1
    x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
    y_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)

    for i in range(0, len(imagelist)):
        print("frame ", imagelist[i])
        x_batch[0] = read_image(imagelist[i], size, offset)
        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))
        loss.unchain_backward()
        loss = 0
        if gpu >= 0: model.to_cpu()
        #write_image(x_batch[0].copy(), 'result/test_' + str(i) + 'x.png')

        print("n ", (i+1)%skip_save_frames)
        if ((i+1)%skip_save_frames == 0):
            num = str(i/skip_save_frames).zfill(10)
            new_filename = output_dir + '/' + num + '.png'
            print("writing ", new_filename)
            write_image(model.y.data[0].copy(), new_filename)

        if gpu >= 0: model.to_gpu()

        # if i == 0 or (args.input_len > 0 and i % args.input_len != 0):
        #     continue
        if i == 0  or (extension_start==0) or (i%extension_start>0):
            continue

        if gpu >= 0: model.to_cpu()
        x_batch[0] = model.y.data[0].copy()
        if gpu >= 0: model.to_gpu()
        print(extension_duration)
        for j in range(0,extension_duration):
            print('extended frameNo:' + str(j + 1))
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            if j == extension_duration.ext - 1:
                g = c.build_computational_graph([model.y])
                node_name = NodeName(g.nodes)
                for n in g.nodes:
                    if isinstance(n, chainer.variable.VariableNode) and \
                      not isinstance(n._variable(), chainer.Parameter) and n.data is not None:
                        img = utils.make_grid(np.expand_dims(chainer.cuda.to_cpu(n.data[-1, ...]), 1))
                        writer.add_image(node_name.name(n), img, i)
            loss.unchain_backward()
            loss = 0
            if gpu >= 0:model.to_cpu()
            write_image(model.y.data[0].copy(), output_dir + '/test_' + str(i) + 'y_' + str(j + 1) + '.png')
            x_batch[0] = model.y.data[0].copy()
            if gpu >= 0:model.to_gpu()
        prednet.reset_state()


# sequencelist = [images_path]
def test_prednet(initmodel, images_list, size, channels, gpu, output_dir = "result", 
                skip_save_frames=0, extension_start=0, extension_duration=0, offset = [0,0], root = "."):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Create Model
    prednet = net.PredNet(size[0], size[1], channels)
    model = L.Classifier(prednet, lossfun=mean_squared_error)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if gpu >= 0:
        cuda.check_cuda_available()
        xp = cuda.cupy
        cuda.get_device(gpu).use()
        model.to_gpu()
        print('Running on GPU')
    else:
        xp = np
        print('Running on CPU')

    # Init/Resume
    serializers.load_npz(initmodel, model)

    test_image_list(prednet, images_list, model, output_dir, channels, size, offset,
                    gpu, skip_save_frames, extension_start, extension_duration)



def train_prednet(initmodel, sequencelist, gpu, size, channels, offset, resume,
                bprop, root,
                output_dir = "result", period=1000000, save=10000):
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Create Model
    prednet = net.PredNet(size[0], size[1], channels)
    model = L.Classifier(prednet, lossfun=mean_squared_error)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if gpu >= 0:
        cuda.check_cuda_available()
        xp = cuda.cupy
        cuda.get_device(gpu).use()
        model.to_gpu()
        print('Running on GPU')
    else:
        xp = np
        print('Running on CPU')

    # Init/Resume
    if initmodel:
        print('Load model from', initmodel)
        serializers.load_npz(initmodel, model)
    if resume:
        print('Load optimizer state from', resume)
        serializers.load_npz(resume, optimizer)

    train_image_folders(sequencelist, prednet, model, channels, size, gpu,
                        period, save, bprop, root)      
    
      
def string_to_intarray(string_input):
    array = string_input.split(',')
    for i in range(len(array)):
        array[i] = int(array[i])

    return array

def call_prednet(args, output_dir = "result"):
    if (not args.images_path) and (not args.sequences):
        print('Please specify images or sequences')
        exit()

    size = string_to_intarray(args.size)
    channels = string_to_intarray(args.channels)
    offset = string_to_intarray(args.offset)

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    if args.images:
        sequencelist = [args.images_path]
    else:
        sequencelist = load_list(args.sequences, args.root)

    if args.test == True:
        test_prednet(args.init_model, sequencelist, output_dir, size, channels, args.gpu,
                            output_dir, args.skip_save_frames, args.ext_t, args.ext, offset, args.root)
    else:
        train_prednet(args.init_model, sequencelist, args.gpu, size, channels,
                            offset, args.resume, args.bprop, args.root, output_dir, args.period, args.save)      

    # For logging graph structure
    model(chainer.Variable(xp.asarray(x_batch)),
          chainer.Variable(xp.asarray(y_batch)))
    writer.add_graph(model.y)
    writer.close()