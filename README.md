# Neural networks for myself
Codes of myself, by myself, for myself.

(......for our study meetings in my laboratory.)
##Required libraries
Python 3.5.4

chainer 5.0.0

pandas 0.20.3

numpy 1.13.1

matplotlib 2.0.2

(As of January 16, 2019)
##Description

I referenced network structures included in "Neural Network Console" .
(Because it is used in our study meeting.)

###auto_encoder

An autoencoder is a model that outputs data same as input data.

In this program, the network structure contains 2 fully connected layers and 2 sigmoid layers.
The loss function is mean squared error.

The model trained with mnist images(4 and 9 only) is in "models" folder in default
(Note: Once you run any train programs, model files will be overwritten).

To train autoencoder, just run auto_encoder.py.
Trained model is saved in "models" folder.
You can visualize outputs through running auto_encoder_visualize.py.

###residual_learning

In this program, I used the network named "Residual network".
In Residual network, there are "residual blocks", and this block has a "shortcut".
There are 2 data path in the block. In the one, do nothing to input data.
In the other one, apply convolution and batch normalization several times, and finally, combine 2 data to 1.
By this technique, we can train deeper neural network.

If you want to know more, please see: [Residual network paper](https://arxiv.org/abs/1512.03385)

In my program, the network has 19 convolution layers, and 19 batch normalization layers, and 1 fully connected layer.
In these layers, 12 convolution layers and 12 batch normalization layers are summarized to 4 "residual blocks".

To train the network, just run residual_learning.py.Trained model is saved in "models" folder.

The model trained with mnist is in "models" folder.