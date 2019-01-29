from chainer import Link, Chain, ChainList
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import math, random
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import link_binary_convolution
import link_binary_linear

IMAGE_SIZE = 28
NUMBER_CLASS = 10


# define loading dataset function
def load_dataset(dataset_path, image_path):
    dataset_csv = pd.read_csv(dataset_path)
    image_path_list = dataset_csv['x:image']
    image_path_list = image_path_list.values.tolist()
    image_label_list = dataset_csv['y:label']
    image_label_list = image_label_list.values.tolist()

    image_list = []
    for path in image_path_list:
        img = cv2.imread((image_path+path.lstrip('.')), cv2.IMREAD_GRAYSCALE)/255.0
        img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
        image_list.append(img)

    return image_list, image_label_list


# define network
class BinaryConnectMnistLeNet(Chain):

    def __init__(self):
        super(BinaryConnectMnistLeNet, self).__init__()
        with self.init_scope():
            self.bconv0 = link_binary_convolution.BinaryConvolution2D(1, 64, ksize=5, pad=0, stride=1)
            self.bn0 = L.BatchNormalization(64)
            self.bconv1 = link_binary_convolution.BinaryConvolution2D(64, 64, ksize=5, pad=0, stride=1)
            self.bn1 = L.BatchNormalization(64)
            self.bfc0 = link_binary_linear.BinaryLinear(1024, 512)
            self.bn2 = L.BatchNormalization(512)
            self.bfc1 = link_binary_linear.BinaryLinear(512, 10)
            self.bn3 = L.BatchNormalization(10)

    def __call__(self, x):
        # x:input
        h = self.bconv0(x)  # BinaryConnectConvolution layer
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)  # MaxPooling layer
        h = self.bn0(h)  # BatchNormalization layer
        h = F.relu(h)  # ReLU layer
        h = self.bconv1(h)  # BinaryConnectConvolution_2 layer
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)  # MaxPooling_2 layer
        h = self.bn1(h)  # BatchNormalization_2 layer
        h = F.relu(h)  # ReLU_3 layer
        h = self.bfc0(h)  # BinaryConnectAffine layer
        h = self.bn2(h)  # BatchNormalization_3 layer
        h = F.relu(h)  # ReLU_2 layer
        h = self.bfc1(h)  # BinaryConnectAffine_2 layer
        h = self.bn3(h)  # BatchNormalization_4 layer

        return h


def main(use_gpu=-1):
    start_time = time.clock()
    # select processing unit
    if use_gpu >= 0:
        import cupy as cp
        xp = cp
        chainer.cuda.get_device(use_gpu).use()
    else:
        xp = np
    # paths set
    training_dataset_path = './samples/sample_dataset/mnist/mnist_training.csv'
    validation_dataset_path = './samples/sample_dataset/mnist/mnist_test.csv'
    image_path = './samples/sample_dataset/mnist'
    # setup network
    model = BinaryConnectMnistLeNet()
    if use_gpu >= 0:
        model.to_gpu()
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=10**(-8), weight_decay_rate=0)
    optimizer.setup(model)

    # setup dataset(training)
    train_image_list, train_image_label_list = load_dataset(training_dataset_path, image_path)

    # setup dataset(validation)
    validation_image_list, validation_image_label_list = load_dataset(validation_dataset_path, image_path)

    epoch = 100
    batchsize = 64
    accuracy_train_list, accuracy_val_list = [], []

    # learning
    for ep in range(0, epoch):
        print('epoch', ep+1)
        # before learning, we have to shuffle training data because we want to make network learn different order for
        # each epoch.
        zipped_train_list = list(zip(train_image_list, train_image_label_list))
        random.shuffle(zipped_train_list)
        learn_image_list, learn_label_list = zip(*zipped_train_list)
        learn_image_list = xp.array(list(learn_image_list))
        learn_label_list = xp.array(list(learn_label_list))
        batch_times = 0
        accuracy_train = 0
        for b in range(0, len(learn_image_list), batchsize):
            model.cleargrads()
            x = chainer.Variable(xp.asarray(learn_image_list[b:b + batchsize]).astype(xp.float32))
            y = chainer.Variable(xp.asarray(learn_label_list[b:b + batchsize]).astype(xp.int32))
            h = model(x)
            # CategorialCrossEntropy doesn't exist in chainer, so, instead of it, I use softmax_cross_entropy.
            loss = F.softmax_cross_entropy(h, y)
            accuracy_train += F.accuracy(h, y).data
            batch_times += 1
            loss.backward()
            optimizer.update()

        accuracy_train_list.append(1-(accuracy_train/batch_times))

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x_valid = chainer.Variable(xp.asarray(validation_image_list).astype(xp.float32))
            y_valid_acc = chainer.Variable(xp.asarray(validation_image_label_list).astype(xp.int32))
            h_valid = model(x_valid)
            accuracy_val = F.accuracy(h_valid, y_valid_acc)

        accuracy_val_list.append(1-accuracy_val.data)

    serializers.save_npz('./models/binary_connect_mnist_LeNet', model)
    print("Time to finish learning:" + str(time.clock() - start_time))
    # draw accuracy graph

    axis_x = np.arange(0, epoch, 1)
    y0 = accuracy_train_list
    y1 = accuracy_val_list
    plt.plot(axis_x, y0, label='train')
    plt.plot(axis_x, y1, label='validation')
    plt.title('Learning Curve', fontsize=20)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Error rate')
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()