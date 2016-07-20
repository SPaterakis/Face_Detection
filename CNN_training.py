"""
===============================================================================
Convolutional Neural Network Training
===============================================================================

Train and fit a single, multi-layered Convolutional Neural
Network on the face and nonface sample generated by 
"generate_samples.py".


Justine Elizabeth Morgan (jem2268)
Stamatios Paterakis (sp3290)
Lauren Nicole Valdivia (lnv2107)
Team ID: 201512-53

Columbia University E6893 Big Data Analytics Fall 2015 Final Project

"""

import numpy as np
import os, sys, scipy
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageOps
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

TrainSet = '...'
CNN_Weights = '...'

def load_data(TrainSet):

    train_x = []
    train_y = []

    # Load faces
    for root, dirs, files in os.walk(os.path.join(TrainSet,'face')):

        for name in files:

            ext = ['.jpg', '.jpeg', '.gif', '.png']

            if name.endswith(tuple(ext)):

                path = os.path.join(root,name)
                image = Image.open(path)
                train_x.append(np.expand_dims(np.asarray(image, dtype='float64')/255, axis=0))
                train_y.append(np.asarray(1, "int32"))

    # Load non-faces
    for root, dirs, files in os.walk(os.path.join(TrainSet,'nonface')):

        for name in files:

            ext = ['.jpg', '.jpeg', '.gif', '.png']

            if name.endswith(tuple(ext)):

                path = os.path.join(root,name)
                image = Image.open(path)
                train_x.append(np.expand_dims(np.asarray(image, dtype='float64')/255, axis=0))
                train_y.append(np.asarray(0, "int32"))

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_x, train_y = shuffle(train_x,train_y, random_state=41)  # shuffle train data

    return train_x, train_y

train_x, train_y = load_data(TrainSet)
w, h = 20, 20  # w x h sized window

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv', layers.Conv2DLayer),
        ('pool', layers.MaxPool2DLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, 1, 20, 20),
    conv_num_filters = 32, conv_filter_size = (3, 3), 
    pool_pool_size = (2, 2),
	hidden_num_units = 50,
    output_num_units = 2, output_nonlinearity = softmax,

    update_learning_rate=0.01,
    update_momentum = 0.9,

    regression = False,
    max_epochs = 60,
    verbose = 1,
    )

net.fit(train_x, train_y)	
net.save_params_to(CNN_Weights) 

train_loss = np.array([i["train_loss"] for i in net.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1.5e-1)
pyplot.yscale("log")
pyplot.show()