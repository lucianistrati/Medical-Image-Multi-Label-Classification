from src.train_neural_net import get_recurrent_classifier, get_conv_classifier, get_fully_connected_classifier

import tensorflow as tf
import numpy as np


sample_datapoint = np.load(file="sample_datapoint.npy", allow_pickle=True)

X_train = np.array([sample_datapoint])
num_classes = 2

options = ["conv", "recurrent", "fully_connected"]
backbone_options_dict = {"conv": [1,2,3,4,5], "recurrent":[1], "fully_connected":[1, 2]}
activation_fn = "sigmoid"

for option in options:
    backbone_options = backbone_options_dict[option]
    for backbone_option in backbone_options:
        if option == "conv":
            input_shape = (X_train[0].shape)
            classifier = get_conv_classifier(num_classes, activation_fn, input_shape, option=backbone_option)
        elif option == "recurrent":
            X_train = X_train[0].reshape((X_train[0].shape[0], X_train[0].shape[1], X_train[0].shape[2]))
            input_shape = (X_train[0].shape)
            classifier = get_recurrent_classifier(num_classes, input_shape, option=backbone_option)
        elif option == "fully_connected":
            input_shape = (X_train[0].shape)
            classifier = get_fully_connected_classifier(num_classes, activation_fn, input_shape, option=backbone_option)
        else:
            raise Exception("Wrong option!")
        dot_img_file = f"arhitectures/{option}_{backbone_option}.png"
        # visualizing how the neural network architectures looks like
        tf.keras.utils.plot_model(classifier, to_file=dot_img_file, show_shapes=True)
