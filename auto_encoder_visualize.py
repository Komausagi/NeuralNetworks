from auto_encoder import load_dataset, Autoencoder
from chainer import cuda
import chainer
import numpy as np
import cv2

IMAGE_SIZE = 28


def main(use_gpu=0):
    if use_gpu >= 0:
        import cupy as cp
        xp = cp
        chainer.cuda.get_device(use_gpu).use()
    else:
        xp = np

    # path set
    training_dataset_path = './samples/sample_dataset/mnist/small_mnist_4or9_training.csv'
    image_path = './samples/sample_dataset/mnist'

    # load data
    visualize_image_list = load_dataset(training_dataset_path, image_path)

    model = Autoencoder()
    chainer.serializers.load_npz('./models/auto_encoder', model)
    model.to_gpu()
    cv2.imshow('Before', visualize_image_list[0].reshape(IMAGE_SIZE, IMAGE_SIZE))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x = chainer.Variable(xp.asarray(visualize_image_list[0]).astype(xp.float32))
        h = model(x)
        h = h.data.reshape(IMAGE_SIZE, IMAGE_SIZE)
    h = cuda.to_cpu(h)
    cv2.imshow('After', h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()