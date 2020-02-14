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

# return the sorted list of images in that folder
def make_list(images_dir):
    temp_list = sorted(os.listdir(images_dir))
    image_list = [os.path.join(images_dir, im)  for im in temp_list]
    return image_list

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

def save_model(count, model, optimizer):
    print('save the model')
    serializers.save_npz('models/' + str(count) + '.model', model)
    print('save the optimizer')
    serializers.save_npz('models/' + str(count) + '.state', optimizer)
    for name, param in model.predictor.namedparams():
        writer.add_histogram(name, chainer.cuda.to_cpu(param.data), count)
    writer.add_scalar('loss', float(model.loss.data), count)

def train_image_list(imagelist, model, optimizer, channels, size, gpu, period, save, bprop, step = 0):
    if len(imagelist) == 0:
        print("Not found images.")
        return

    logf = open('log.txt', 'w')

    xp = cuda.cupy if gpu >= 0 else np
    batchSize = 1
    x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
    y_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)

    x_batch[0] = read_image(imagelist[0], size, offset)
    loss = 0
    for i in range(1, len(imagelist)):
        y_batch[0] = read_image(imagelist[i], size, offset);
        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))

        if (i + 1) % bprop == 0:
            print("count ", step," frameNo ", i)
            model.zerograds()
            loss.backward()
            loss.unchain_backward()
            loss = 0
            optimizer.update()
            # if gpu >= 0:model.to_cpu()
            # write_image(x_batch[0].copy(), 'images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'x.png')
            # write_image(model.y.data[0].copy(), 'images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'y.png')
            # write_image(y_batch[0].copy(), 'images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'z.png')
            if gpu >= 0:model.to_gpu()
            print('loss:' + str(float(model.loss.data)))
            logf.write(str(i) + ', ' + str(float(model.loss.data)) + '\n')

        step += 1
        if (step%save) == 0:
            save_model(step, model, optimizer)
        x_batch[0] = y_batch[0]
        
        if (step>=period):
            break

    return step


def train_image_folders(sequencelist, prednet, model, optimizer,
                        channels, size, gpu, period, save, bprop):

    step = 0
    while step<period:
        for sequence in sequencelist:
            prednet.reset_state()
            imagelist = make_list(sequence)
            step = train_image_list(imagelist, model, optimizer, channels, size, gpu, 
                        period, save, bprop, step)
            if (step>=period):
                break

    save_model(step, model, optimizer)


def test_image_list(prednet, imagelist, model, output_dir, channels, size, offset, gpu, skip_save_frames=0, 
    extension_start=0, extension_duration=100, reset_each = False):

    xp = cuda.cupy if gpu >= 0 else np

    prednet.reset_state()
    loss = 0
    batchSize = 1
    x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
    y_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)

    for i in range(0, len(imagelist)):
        # print("frame ", imagelist[i])
        x_batch[0] = read_image(imagelist[i], size, offset)
        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))
        loss.unchain_backward()
        loss = 0
        if gpu >= 0: model.to_cpu()
        #write_image(x_batch[0].copy(), 'result/test_' + str(i) + 'x.png')

        if ((i+1)%skip_save_frames == 0):
            num = str(i//skip_save_frames).zfill(10)
            new_filename = output_dir + '/' + num + '.png'
            print("writing ", new_filename)
            write_image(model.y.data[0].copy(), new_filename)

        if gpu >= 0: model.to_gpu()
        if reset_each:
            prednet.reset_state()

        if i == 0  or (extension_start==0) or (i%extension_start>0):
            continue

        if gpu >= 0: model.to_cpu()
        x_batch[0] = model.y.data[0].copy()
        if gpu >= 0: model.to_gpu()

        for j in range(0,extension_duration):
            print('extended frameNo:' + str(j + 1))
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            if j == extension_duration - 1:
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


# sequence_list = [path, path] of folders with text file listing images
def test_prednet(initmodel, sequence_list, size, channels, gpu, output_dir="result", 
                skip_save_frames=0, extension_start=0, extension_duration=0, offset = [0,0], reset_each = False):

    #Create Model
    prednet = net.PredNet(size[0], size[1], channels)
    model = L.Classifier(prednet, lossfun=mean_squared_error)
    model.compute_accuracy = False
    # optimizer = optimizers.Adam()
    # optimizer.setup(model)

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

    for seq in sequence_list:
        image_list = make_list(seq)
        test_image_list(prednet, image_list, model, output_dir, channels, size, offset,
                        gpu, skip_save_frames, extension_start, extension_duration, reset_each)



def train_prednet(initmodel, sequencelist, gpu, size, channels, offset, resume,
                bprop, output_dir="result", period=1000000, save=10000):
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('images'):
        os.makedirs('images')

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

    train_image_folders(sequencelist, prednet, model, optimizer, 
                        channels, size, gpu, period, save, bprop)   

    # # For logging graph structure
    # model(chainer.Variable(xp.asarray(x_batch)),
    #       chainer.Variable(xp.asarray(y_batch)))
    # writer.add_graph(model.y)
    # writer.close()
      
def string_to_intarray(string_input):
    array = string_input.split(',')
    for i in range(len(array)):
        array[i] = int(array[i])

    return array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='PredNet')
    parser.add_argument('--images_path', '-i', default='', help='Path input images')
    parser.add_argument('--output_dir', '-out', default= "result", help='where to save predictions')
    parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--size', '-s', default='160,120',
                        help='Size of target images. width,height (pixels)')
    parser.add_argument('--channels', '-c', default='3,48,96,192',
                        help='Number of channels on each layers')
    parser.add_argument('--offset', '-off', default='0,0',
                        help='Center offset of clipping input image (pixels)')
    # parser.add_argument('--input_len', '-l', default=50, type=int,
    #                     help='Input frame length fo extended prediction on test (frames)')
    parser.add_argument('--ext', '-e', default=0, type=int,
                        help='Extended prediction on test (frames)')
    parser.add_argument('--ext_t', default=20, type=int,
                        help='When to start extended prediction')
    parser.add_argument('--bprop', default=20, type=int,
                        help='Back propagation length (frames)')
    parser.add_argument('--save', default=10000, type=int,
                        help='Period of save model and state (frames)')
    parser.add_argument('--period', default=1000000, type=int,
                        help='Period of training (frames)')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--skip_save_frames', '-sikp', type=int, default=1, help='predictions will be saved every x steps')

    parser.set_defaults(test=False)
    args = parser.parse_args()

    if (not args.images_path) and (not args.sequences):
        print('Please specify images or sequences')
        exit()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    size = string_to_intarray(args.size)
    channels = string_to_intarray(args.channels)
    offset = string_to_intarray(args.offset)

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    if args.images_path:
        sequencelist = [args.images_path]
    else:
        sequencelist = args.sequences

    if args.test == True:
        test_prednet(args.initmodel, sequencelist, size, channels, args.gpu, args.output_dir,
                    args.skip_save_frames, args.ext_t, args.ext, offset)
    else:
        train_prednet(args.initmodel, sequencelist, args.gpu, size, channels,
                            offset, args.resume, args.bprop, args.output_dir, args.period, args.save)  


