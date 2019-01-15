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

IMAGE_SIZE = 28


# define loading dataset function
def load_dataset(dataset_path, image_path):
    dataset_csv = pd.read_csv(dataset_path)
    image_path_list = dataset_csv['x:image']
    image_path_list = image_path_list.values.tolist()
    image_list = []

    for path in image_path_list:
        img = cv2.imread((image_path+path.lstrip('.')), cv2.IMREAD_GRAYSCALE)/255.0
        img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
        image_list.append(img)

    return image_list


# define network
class Autoencoder(Chain):

    def __init__(self):
        super(Autoencoder, self).__init__()
        with self.init_scope():
            self.fc0 = L.Linear(None, 256)
            self.fc1 = L.Linear(None, 784)

    def __call__(self, x):
        h = F.dropout(x, ratio=0.5)
        h = self.fc0(h)  # Affine_1 layer
        h = F.sigmoid(h) # Sigmoid_1 layer
        h = self.fc1(h) # Affine_2 layer
        h = F.sigmoid(h)  # Sigmoid_2 layer
        return h


def main(use_gpu=0):
    start_time = time.clock()
    # select processing unit
    if use_gpu >= 0:
        import cupy as cp
        xp = cp
        chainer.cuda.get_device(use_gpu).use()
    else:
        xp = np
    # paths set
    training_dataset_path = './samples/sample_dataset/mnist/small_mnist_4or9_training.csv'
    validation_dataset_path = './samples/sample_dataset/mnist/small_mnist_4or9_test.csv'
    image_path = './samples/sample_dataset/mnist'
    # setup network
    model = Autoencoder()
    if use_gpu >= 0:
        model.to_gpu()
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=10**(-8), weight_decay_rate=0)
    optimizer.setup(model)

    # setup dataset(training)
    train_image_list = load_dataset(training_dataset_path, image_path)

    # setup dataset(validation)
    validation_image_list = load_dataset(validation_dataset_path, image_path)

    epoch = 100
    batchsize=64
    loss_train_list, loss_val_list = [], []
    batch_times = 0
    # learning
    for ep in range(0, epoch):
        print('epoch', ep+1)
        # before learning, we have to shuffle training data because we want to make network learn different order for
        # each epoch.
        random.shuffle(train_image_list)
        loss_sum = 0
        for b in range(0, len(train_image_list), batchsize):
            model.cleargrads()
            x = chainer.Variable(xp.asarray(train_image_list[b:b + batchsize]).astype(xp.float32))
            h = model(x)
            # squared error
            loss = F.mean_squared_error(h.reshape(len(x), 1, IMAGE_SIZE, IMAGE_SIZE), x)
            loss_sum += loss.data
            batch_times += 1
            loss.backward()
            optimizer.update()

        loss_train_list.append(loss_sum/batch_times)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x_valid = chainer.Variable(xp.asarray(validation_image_list).astype(xp.float32))
            h_valid = model(x_valid)
            loss_val = F.mean_squared_error(h_valid.reshape(len(x_valid), 1 , IMAGE_SIZE, IMAGE_SIZE), x_valid)

        loss_val_list.append(loss_val.data)

    serializers.save_npz('./models/auto_encoder', model)
    print("Time to finish learning:" + str(time.clock() - start_time))
    # draw accuracy graph
    axis_x = np.arange(0, epoch, 1)
    y0 = loss_train_list
    y1 = loss_val_list
    plt.plot(axis_x, y0, label='train')
    plt.plot(axis_x, y1, label='validation')
    plt.title('Learning Curve', fontsize=20)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Loss')
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()