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
# from tb_chainer import SummaryWriter, NodeName, utils
from . import net

# return the sorted list of images in that folder
def make_list(images_dir, limit):
    temp_list = sorted(os.listdir(images_dir))
    if(limit>0):
        temp_list = temp_list[0:limit]
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

# writer = SummaryWriter('runs/test')#+datetime.now().strftime('%B%d  %H:%M:%S'))

def save_model(count, model, optimizer):
    print('save the model')
    serializers.save_npz('models/' + str(count) + '.model', model)
    print('save the optimizer')
    serializers.save_npz('models/' + str(count) + '.state', optimizer)
    # for name, param in model.predictor.namedparams():
    #     writer.add_histogram(name, chainer.cuda.to_cpu(param.data), count)
    # writer.add_scalar('loss', float(model.loss.data), count)


def train_image_list(imagelist, model, optimizer, channels, size, offset, gpu, period, save, 
                     bprop, logf, step = 0, verbose = 1):

    if len(imagelist) == 0:
        print("Not found images.")
        return

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

        if (step + 1) % bprop == 0:
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
            if verbose == 1:
                print("step ", step," frame ", i, "loss:", model.loss.data)
            logf.write(str(step) + ', ' + str(float(model.loss.data)) + '\n')
            logf.flush()

        step += 1
        if (step%save) == 0:
            save_model(step, model, optimizer)
        x_batch[0] = y_batch[0]
        
        if (step>=period):
            break

    return step


def train_image_sequences(sequence_list, prednet, model, optimizer,
                        channels, size, offset, gpu, period, save, bprop, verbose = 1):
    step = 0
    logf = open('train_log.txt', 'w')
    while step<period:
        for image_list in sequence_list:
            prednet.reset_state()
            step = train_image_list(image_list, model, optimizer, channels, size, offset, gpu, 
                        period, save, bprop, logf, step, verbose)
            if (step>=period):
                break

    save_model(step, model, optimizer)


# imagelist = [path, path, path]
def test_image_list(prednet, imagelist, model, output_dir, channels, size, offset, gpu, logf, skip_save_frames=0, 
    extension_start=0, extension_duration=100, reset_each = False, step = 0, verbose = 1, reset_at = -1):

    xp = cuda.cupy if gpu >= 0 else np

    prednet.reset_state()
    loss = 0
    batchSize = 1
    x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
    y_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)

    if reset_each:
        reset_at = 1

    for i in range(0, len(imagelist)):
        x_batch[0] = read_image(imagelist[i], size, offset)
        if(i<len(imagelist)-1):
            y_batch[0] = read_image(imagelist[i+1], size, offset)

        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))
        loss.unchain_backward()
        loss = 0
        if gpu >= 0: model.to_cpu()

        if(i<len(imagelist)-1):
            if verbose == 1:
                print("step ", step," frame ", i, "loss:", model.loss.data)
            logf.write(str(step) + ', ' + str(float(model.loss.data)) + '\n')
            logf.flush()
        else:
            if verbose == 1:
                print("step ", step," frame ", i, "loss: last frame.")

        if ((step+1)%skip_save_frames == 0):
            num = str(step//skip_save_frames).zfill(10)
            new_filename = output_dir + '/' + num + '.png'
            if verbose == 1:
                print("writing ", new_filename)
            write_image(model.y.data[0].copy(), new_filename)

        if gpu >= 0: model.to_gpu()

        step = step + 1
        #print("step", step)
        if step == 0  or (extension_start==0) or (step%extension_start>0):
            if reset_at>0 and i%reset_at==0:
                prednet.reset_state()
            continue

        if gpu >= 0: model.to_cpu()
        x_batch[0] = model.y.data[0].copy()
        if gpu >= 0: model.to_gpu()

        for j in range(0,extension_duration):

            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            # if j == extension_duration - 1:
            #     g = c.build_computational_graph([model.y])
            #     node_name = NodeName(g.nodes)
            #     for n in g.nodes:
            #         if isinstance(n, chainer.variable.VariableNode) and \
            #           not isinstance(n._variable(), chainer.Parameter) and n.data is not None:
            #             img = utils.make_grid(np.expand_dims(chainer.cuda.to_cpu(n.data[-1, ...]), 1))
            #             writer.add_image(node_name.name(n), img, i)
            loss.unchain_backward()
            loss = 0
            if gpu >= 0:model.to_cpu()
            num = str(step//skip_save_frames + j ).zfill(10)
            new_filename = output_dir + '/' + num + '_extended.png'
            if verbose == 1:
                print("writing ", new_filename)

            write_image(model.y.data[0].copy(), new_filename)
            x_batch[0] = model.y.data[0].copy()
            if gpu >= 0:model.to_gpu()
        prednet.reset_state()

    return step


# sequence_list = [[path,path,path], [path,path,path]] list of lists of images
def test_prednet(initmodel, sequence_list, size, channels, gpu, output_dir="result", 
                skip_save_frames=0, extension_start=0, extension_duration=0, offset = [0,0], 
                reset_each = False, verbose = 1, reset_at = -1):

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

    logf = open('test_log.txt', 'w')
    step = 0
    if verbose == 1:
        print("sequence_list ", sequence_list)
    for image_list in sequence_list:
        step = test_image_list(prednet, image_list, model, output_dir, channels, size, offset,
                                gpu, logf, skip_save_frames, extension_start, extension_duration,
                                reset_each, step, verbose, reset_at)
        

def train_prednet(initmodel, sequence_list, gpu, size, channels, offset, resume,
                bprop, output_dir="result", period=1000000, save=10000, verbose = 1):
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

    train_image_sequences(sequence_list, prednet, model, optimizer, 
                        channels, size, offset, gpu, period, save, bprop, verbose)   

      
def string_to_intarray(string_input):
    array = string_input.split(',')
    for i in range(len(array)):
        array[i] = int(array[i])

    return array

def call_with_args(args):  
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
        temp_list = make_list(args.images_path, args.input_len)
        sequence_list = [temp_list]
    else:
        # read file
        temp_list = [line.rstrip('\n') for line in open(args.sequences)]
        # now read files in list
        # extract path of seq list file
        array = args.sequences.split('/')
        base_path = os.path.abspath(os.getcwd()) + "/" + '/'.join(array[:-1])
        sequence_list = [None]* len(temp_list)
        i = 0
        for path in temp_list:
            sequence_list[i] = [os.path.join(base_path,line.rstrip('\n')) for line in open(os.path.join(base_path,path))]
            i = i+1

    if args.test == True:
        test_prednet(args.initmodel, sequence_list, size, channels, args.gpu, args.output_dir,
                    args.skip_save_frames, args.ext_t, args.ext, offset, args.verbose)
    else:
        train_prednet(args.initmodel, sequence_list, args.gpu, size, channels,
                            offset, args.resume, args.bprop, args.output_dir, args.period, args.save, args.verbose)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='PredNet')
    parser.add_argument('--images_path', '-i', default='', help='Path to input images')
    parser.add_argument('--output_dir', '-out', default= "result", help='where to save predictions')
    parser.add_argument('--sequences', '-seq', default='', help='In text mode, Path to file with list of text files, that themselves contain lists of images')
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
    parser.add_argument('--input_len', default=-1, type=int,
                        help='how many frames to use if using images_path')
    parser.add_argument('--verbose', '-v', default=1, type=int,
                        help='Output progression logs (1) or not (0)')

    parser.set_defaults(test=False)
    args = parser.parse_args()

    call_with_args(args)

