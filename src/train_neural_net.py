from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, \
    recall_score, average_precision_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import check_array

import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def empty_classif_loggings():
    return [['F1', 0.0], ['Accuracy', 0.0], ['Normalized_confusion_matrix',
                                             0.0], ["Precision", 0.0], ["average_precision_score", 0.0],
            ["Recall", 0.0]]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def get_classif_perf_metrics(y_test, y_pred, model_name="",
                             logging_metrics_list=empty_classif_loggings()):
    for model_categoy in ["FeedForward", "Convolutional", "LSTM"]:
        if model_categoy in model_name:
            y_pred = np.array([np.argmax(pred) for pred in y_pred])
            y_test = np.array([np.argmax(pred) for pred in y_test])
    print("For " + model_name + " classification algorithm the following "
                                "performance metrics were determined on the "
                                "test set:")
    number_of_classes = max(y_test) + 1
    print("NUM CLASSES", number_of_classes)
    if number_of_classes == 2:
        for i in range(len(logging_metrics_list)):
            if logging_metrics_list[i][0] == 'Accuracy':
                logging_metrics_list[i][1] = str(accuracy_score(y_test, y_pred))
            elif logging_metrics_list[i][0] == 'Precision':
                logging_metrics_list[i][1] = str(precision_score(y_test,
                                                                 y_pred))
            elif logging_metrics_list[i][0] == 'Recall':
                logging_metrics_list[i][1] = str(recall_score(y_test, y_pred))
            elif logging_metrics_list[i][0] == 'F1':
                logging_metrics_list[i][1] = str(f1_score(y_test, y_pred))
            elif logging_metrics_list[i][0] == "average_precision_score":
                logging_metrics_list[i][1] = str(average_precision_score(y_test, y_pred))

        print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
        print("Precision: " + str(precision_score(y_test, y_pred)))
        print("Recall: " + str(recall_score(y_test, y_pred)))
    else:
        for i in range(len(logging_metrics_list)):
            if logging_metrics_list[i][0] == 'Classification_report':
                logging_metrics_list[i][1] = str(
                    classification_report(y_test, y_pred, digits=2))

        print("Classification report: \n" + str(
            classification_report(y_test, y_pred, digits=2)))

    C = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", C)
    print("Normalized confusion matrix:\n",
          np.around(C / C.astype(np.float).sum(axis=1), decimals=2))
    for i in range(len(logging_metrics_list)):
        if logging_metrics_list[i][0] == 'Confusion_matrix':
            logging_metrics_list[i][1] = np.array2string(np.around(C,
                                                                   decimals=2),
                                                         precision=2,
                                                         separator=',',
                                                         suppress_small=True)
        elif logging_metrics_list[i][0] == 'Normalized_confusion_matrix':
            logging_metrics_list[i][1] = np.array2string(np.around(C / C.astype(np.float).sum(axis=1),
                                                                   decimals=2), precision=2,
                                                         separator=',', suppress_small=True)
    return logging_metrics_list


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def get_classif_perf_metrics(y_test, y_pred, model_name="",
                             logging_metrics_list=empty_classif_loggings(), num_classes=1):
    for model_categoy in ["FeedForward", "Convolutional", "LSTM"]:
        if model_categoy in model_name:
            y_pred = np.array([np.argmax(pred) for pred in y_pred])
            y_test = np.array([np.argmax(pred) for pred in y_test])
    print("For " + model_name + " classification algorithm the following "
                                "performance metrics were determined on the "
                                "test set:")
    number_of_classes = num_classes
    print("NUM CLASSES", number_of_classes)

    for i in range(len(logging_metrics_list)):
        if logging_metrics_list[i][0] == 'Accuracy':
            logging_metrics_list[i][1] = str(accuracy_score(y_test, y_pred))
        elif logging_metrics_list[i][0] == 'Precision':
            logging_metrics_list[i][1] = str(precision_score(y_test,
                                                             y_pred))
        elif logging_metrics_list[i][0] == 'Recall':
            logging_metrics_list[i][1] = str(recall_score(y_test, y_pred))
        elif logging_metrics_list[i][0] == 'F1':
            logging_metrics_list[i][1] = str(f1_score(y_test, y_pred,
                                                      average='weighted'))
    print("Accuracy: " + str(round(accuracy_score(y_test, y_pred), 2)))
    print("Precision: " + str(precision_score(y_test, y_pred,
                                              average='weighted')))
    print("Recall: " + str(recall_score(y_test, y_pred,
                                        average='weighted')))

    C = confusion_matrix(y_test, y_pred)

    print("Confusion matrix:\n", C)
    print("Normalized confusion matrix:\n",
          np.around(C / C.astype(np.float).sum(axis=1), decimals=2))

    for i in range(len(logging_metrics_list)):
        if logging_metrics_list[i][0] == 'Confusion_matrix':
            logging_metrics_list[i][1] = np.array2string(np.around(C,
                                                                   decimals=2),
                                                         precision=2,
                                                         separator=',',
                                                         suppress_small=True)
        elif logging_metrics_list[i][0] == 'Normalized_confusion_matrix':
            logging_metrics_list[i][1] = np.array2string(np.around(C / C.astype(np.float).sum(axis=1),
                                                                   decimals=2), precision=2,
                                                         separator=',', suppress_small=True)

    return logging_metrics_list


def plot_loss_and_acc(history):
    plt.plot(history.history["acc"], label="train")
    plt.plot(history.history["val_acc"], label="test")

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["train", "test"])
    plt.show()


def plot_multiple_metrics(history, model_name=""):
    keys = list(history.history.keys())
    colors = ['g', 'b', 'r', 'y', 'p']
    for i in range(len(keys)):
        hist_key = keys[i]
        metric = history.history[hist_key]
        actual_num_epochs = range(1, len(metric) + 1)
        plt.plot(actual_num_epochs, metric, colors[i], label=hist_key)
    if model_name:
        plt.title("Metrics obtained for " + model_name)
    plt.legend()
    plt.show()


def plot_metric(history, metric_name, model_name):
    if "acc" in history.history.keys() or "accuracy" in history.history.keys():
        if "accuracy" in history.history.keys():
            metric = history.history["accuracy"]
            val_metric = history.history["val_accuracy"]
        else:
            metric = history.history[metric_name]
            val_metric = history.history["val_" + metric_name]
    else:
        metric = history.history[metric_name]
        val_metric = history.history["val_" + metric_name]

    actual_num_epochs = range(1, len(metric) + 1)

    plt.plot(actual_num_epochs, metric, "g",
             label="Train " + metric_name + " for " + model_name)
    plt.plot(actual_num_epochs, val_metric, "b",
             label="Val " + metric_name + " for " + model_name)
    plt.legend()
    plt.title(metric_name.capitalize() + " for " + model_name)
    plt.show()


def get_conv_classifier(num_classes, activation_fn, input_shape, option: int = 1):
    """
    https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418
    to be tryed out TODO!!!!!!!!111 BUT THERE ARE PICS WITH code, not code
    """
    model = tf.keras.models.Sequential()

    if option == 1:
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(64, 64, 1)))
        model.add(tf.keras.layers.MaxPool2D(3, 3))
        # classifier.add(tf.keras.layers.Conv2D(48, (2, 2), activation='relu'))
        # classifier.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation=activation_fn))
    elif option == 2:
        # https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

        model.add(tf.keras.layers.Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))

        model.add(tf.keras.layers.Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=100, activation='relu'))
        model.add(tf.keras.layers.Dense(units=100, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    elif option == 3:
        model.add(tf.keras.layers.Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))

        model.add(tf.keras.layers.Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(250, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Dense(2, activation='softmax'))
    elif option == 4:
        # https://www.geeksforgeeks.org/image-classifier-using-cnn/
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPool2D(5))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.8))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
    elif option == 5:
        # https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                         activation='relu', input_shape=input_shape))

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv2D(512, (5, 5), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(512, (5, 5), activation='relu'))

        model.add(tf.keras.layers.MaxPooling2D((4, 4), padding="same"))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


def get_fully_connected_classifier(num_classes, activation_fn, input_shape=None, option: int = 1):
    classifier = tf.keras.models.Sequential()

    if option == 1:
        classifier.add(tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape))
        classifier.add(tf.keras.layers.Dense(256, activation='relu'))
        # classifier.add(tf.keras.layers.Dense(256, input_shape=X_test[0].shape, activation="relu"))
        # classifier.add(tf.keras.layers.Dense(256, activation="relu"))
        classifier.add(tf.keras.layers.Dense(num_classes, activation=activation_fn))
    elif option == 2:
        classifier.add(tf.keras.layers.Dense(256, input_shape=input_shape))
        classifier.add(tf.keras.layers.Dense(100, activation='relu'))
        classifier.add(tf.keras.layers.Dropout(0.5))
        classifier.add(tf.keras.layers.BatchNormalization())
        classifier.add(tf.keras.layers.Dense(2, activation='softmax'))

    return classifier


def get_recurrent_classifier(num_classes, input_shape=None):
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.LSTM(128, input_shape=input_shape,
                                        return_sequences=True))
    classifier.add(tf.keras.layers.LSTM(128, return_sequences=True))
    classifier.add(tf.keras.layers.Flatten())
    classifier.add(tf.keras.layers.Dense(64, activation="relu"))
    classifier.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    return classifier


def normalize_img(img):
    return img / 255


def train_nn(X_train, y_train, X_test, y_test, model_name, num_classes=2, X_submission=None, class_weight=None):
    early_stopping = EarlyStopping(
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
    )

    lr_schedule = ReduceLROnPlateau(
        patience=0,
        factor=0.2,
        min_lr=0.001,
    )

    n_epochs = [5, 10]
    learning_rate = [1e-3, 1e-2, 1e-1]

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    activation_fn = ["sigmoid", "softmax", "linear"][0]

    option = ["conv", "recurrent", "fully_connected"][0]

    print((option + "    ") * 50)

    if option == "conv":
        input_shape = (X_train[0].shape)
        classifier = get_conv_classifier(num_classes, activation_fn, input_shape)
    elif option == "recurrent":
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
        input_shape = (X_train[0].shape)
        classifier = get_recurrent_classifier(num_classes, input_shape)
    elif option == "fully_connected":
        input_shape = (X_train[0].shape)
        classifier = get_fully_connected_classifier(num_classes, activation_fn, input_shape)
    else:
        raise Exception("Wrong option!")

    opt = [tf.keras.optimizers.Adam(lr=learning_rate), tf.keras.optimizers.SGD(lr=learning_rate)][0]

    loss_function = [tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalCrossentropy()][0]

    metrics_function = [tfa.metrics.F1Score(num_classes=2), "accuracy"][0]

    classifier.compile(optimizer=opt, loss=loss_function, metrics=[metrics_function])

    validation_option = [None, "split", "data"][1]
    validation_split = 0.1
    validation_data = (X_test, y_test)

    if validation_option == "split":
        X_train = np.concatenate((X_train, X_test))
        y_train = np.concatenate((y_train, y_test))

    if class_weight is not None:
        if validation_option is None:
            history = classifier.fit(X_train,
                                     y_train,
                                     epochs=n_epochs,
                                     verbose=2,
                                     class_weight=class_weight,
                                     callbacks=[early_stopping, lr_schedule])
        elif validation_option is "split":
            history = classifier.fit(X_train,
                                     y_train,
                                     epochs=n_epochs,
                                     verbose=2,
                                     class_weight=class_weight,
                                     callbacks=[early_stopping, lr_schedule],
                                     validation_split=validation_split)
        elif validation_option is "data":
            history = classifier.fit(X_train,
                                     y_train,
                                     epochs=n_epochs,
                                     verbose=2,
                                     class_weight=class_weight,
                                     callbacks=[early_stopping, lr_schedule],
                                     validation_data=validation_data)
        else:
            raise Exception(f"Wrong validation_option: {validation_option}")
    else:
        if validation_option is None:
            history = classifier.fit(X_train,
                                     y_train,
                                     epochs=n_epochs,
                                     verbose=2,
                                     callbacks=[early_stopping, lr_schedule])
        elif validation_option is "split":
            history = classifier.fit(X_train,
                                     y_train,
                                     epochs=n_epochs,
                                     verbose=2,
                                     callbacks=[early_stopping, lr_schedule],
                                     validation_data=validation_data)
        elif validation_option is "data":
            history = classifier.fit(X_train,
                                     y_train,
                                     epochs=n_epochs,
                                     verbose=2,
                                     callbacks=[early_stopping, lr_schedule],
                                     validation_split=validation_split)
        else:
            raise Exception(f"Wrong validation_option: {validation_option}")

    logging_metrics_list = None

    if validation_option is None:
        y_pred = classifier.predict(X_test)

        print(classifier.summary())

        if num_classes > 1:
            y_test = np.argmax(y_test, axis=-1)
            y_pred = np.argmax(y_pred, axis=-1)

        if num_classes > 1:
            logging_metrics_list = get_classif_perf_metrics(y_test,
                                                            y_pred,
                                                            model_name=model_name, num_classes=num_classes)

        print(logging_metrics_list)

        plot_multiple_metrics(history)

    y_submission = classifier.predict(X_submission)

    if num_classes > 1:
        y_submission = np.argmax(y_submission, axis=-1)

    return y_submission, logging_metrics_list
