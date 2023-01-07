from tensorflow.keras.layers import Conv1D, MaxPool1D

# from keras.layers.convolutional import AtrousConvolution1D
# from price_logger import log_results_regress, log_results_classif
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, \
    BatchNormalization
# from price_backtester import price_backtest
# from price_genetic_model import price_genetic
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
# from price_metrics import \
    # get_classif_perf_metrics, \
    # get_regress_perf_metrics
from copy import deepcopy
# from price_visualizer import plot_metric, plot_actual_and_predicted_price, \
    # plot_multiple_metrics

import numpy as np


import visualkeras

from src.train_neural_net import get_recurrent_classifier, get_conv_classifier, get_fully_connected_classifier

sample_datapoint = np.load(file="sample_datapoint.npy", allow_pickle=True)
X_train = [sample_datapoint]
num_classes = 2

options = ["conv", "recurrent", "fully_connected"]
backbone_options_dict = {"conv": [1,2,3,4,5], "recurrent":[1], "fully_connected":[1, 2]}
activation_fn = "sigmoid"

classifier = Sequential()
classifier.add(Conv1D(kernel_size=3, filters=24,
                     input_shape=(220, 1)))
classifier.add(MaxPool1D(pool_size=2))
classifier.add(Flatten())
classifier.add(Dense(1))
import tensorflow as tf

dot_img_file = 'model_price.png'
tf.keras.utils.plot_model(classifier, to_file=dot_img_file, show_shapes=True)


for option in options:
    backbone_options = backbone_options_dict[option]
    for backbone_option in backbone_options:
        if option == "conv":
            input_shape = (X_train[0].shape)
            classifier = get_conv_classifier(num_classes, activation_fn, input_shape, option=backbone_option)
        elif option == "recurrent":
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
            input_shape = (X_train[0].shape)
            classifier = get_recurrent_classifier(num_classes, input_shape, option=backbone_option)
        elif option == "fully_connected":
            input_shape = (X_train[0].shape)
            classifier = get_fully_connected_classifier(num_classes, activation_fn, input_shape, option=backbone_option)
        else:
            raise Exception("Wrong option!")
        dot_img_file = f"{option}_{backbone_option}.png"
        tf.keras.utils.plot_model(classifier, to_file=dot_img_file, show_shapes=True)

# visualkeras.layered_view(model).show() # display using your system viewer
# visualkeras.layered_view(model, to_file='output.png') # write to disk
# visualkeras.layered_view(model, to_file='output.png').show() # write and show

# visualkeras.layered_view(model)