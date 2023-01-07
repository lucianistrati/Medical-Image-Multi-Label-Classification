from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import check_array


def empty_classif_loggings():
    return [['F1', 0.0], ['Accuracy', 0.0], ['Normalized_confusion_matrix',
                                             0.0]]


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


def get_conv_classifier(num_classes, activation_fn, input_shape, option: int=1):
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
        # cnn 2 is also 89% smth, as rnn lstm is 89% smth as well
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

        # 1st conv block
        model.add(tf.keras.layers.Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
        # 2nd conv block
        model.add(tf.keras.layers.Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        # 3rd conv block
        model.add(tf.keras.layers.Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))
        model.add(tf.keras.layers.BatchNormalization())
        # ANN block
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=100, activation='relu'))
        model.add(tf.keras.layers.Dense(units=100, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.25))
        # output layer
        model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    elif option == 3:
        model.add(tf.keras.layers.Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))

        # convolutional layer
        model.add(tf.keras.layers.Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # flatten output of conv
        model.add(tf.keras.layers.Flatten())

        # hidden layer
        model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(250, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        # output layer
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
        ### Input Layer ####
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                         activation='relu', input_shape=input_shape))

        ### Convolutional Layers ####
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))  # Pooling
        model.add(tf.keras.layers.Dropout(0.2))  # Dropout

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

        #### Fully-Connected Layer ####
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


def get_fully_connected_classifier(num_classes, activation_fn, input_shape=None):
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape))
    classifier.add(tf.keras.layers.Dense(256, activation='relu'))
    # # classifier.add(tf.keras.layers.Dense(256, input_shape=X_test[0].shape, activation="relu"))
    # # classifier.add(tf.keras.layers.Dense(256, activation="relu"))
    classifier.add(tf.keras.layers.Dense(num_classes, activation=activation_fn))

    # model2.add(Flatten(input_shape=(7, 7, 512)))
    # model2.add(Dense(100, activation='relu'))
    # model2.add(Dropout(0.5))
    # model2.add(BatchNormalization())
    # model2.add(Dense(10, activation='softmax'))

    return classifier


def get_recurrent_classifier(num_classes, activation_fn, input_shape=None):
    #
    # for i in range(len(X_train)):
    #     X_train[i] = X_train[i].reshape((1, X_train[0].shape[0], 1))
    #
    # for i in range(len(X_validation)):
    #     X_validation[i] = X_validation[i].reshape((1, X_validation[0].shape[0],
    #                                                1))
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.LSTM(128, input_shape=input_shape,
                                        return_sequences=True))
    classifier.add(tf.keras.layers.LSTM(128, return_sequences=True))
    classifier.add(tf.keras.layers.Flatten())
    classifier.add(tf.keras.layers.Dense(64, activation="relu"))
    classifier.add(tf.keras.layers.Dense(2, activation="softmax"))

    return classifier


def get_model_design_1(filters: list, input_shape: tuple):
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters[0], (5, 5), padding='same',
                                                               kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                                               activation="relu", input_shape=input_shape),
                                        tf.keras.layers.Conv2D(filters[1], (3, 3), padding='same',
                                                               kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                                               activation="relu"),
                                        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Dense(2,
                                                              kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                              activation="softmax")
                                        ])

    return model


def normalize_img(img):
    return img / 255


def train_nn(X_train, y_train, X_test, y_test, model_name, num_classes=2, X_submission=None, class_weight=None):
    early_stopping = EarlyStopping(
        patience=5,  # how many epochs to wait before stopping
        min_delta=0.001,  # minimium amount of change to count as an improvement
        restore_best_weights=True,
    )

    lr_schedule = ReduceLROnPlateau(
        patience=0,
        factor=0.2,
        min_lr=0.001,
    )

    # TODO try to train in a multi-labels classif way with keras, or pytorch if not possible with keras, perhaps it
    # will learn better in this multimodal format (since the classification might be related one with another
    # oooor, maybe do: first classification, then use that prediction for the second classification and then use the
    # first and the third to also produce the fourth
    # try to visualize data in 2D/3D

    n_epochs = 10  # TODO adapt
    learning_rate = 1e-3  # TODO adapt

    # if num_classes > 1:
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    activation_fn = "linear"
    activation_fn = "softmax"
    activation_fn = "sigmoid"
    # TODO adapt, maybe sigmoid instead of softmax since there are two classes and is not multi class

    # filters = [128, 64]
    # classifier = get_model_design_1(filters, X_train[0].shape)

    option = ["conv", "recurrent", "fully_connected"][0]

    print((option + "    ") * 50)

    if option == "conv":
        input_shape = (X_train[0].shape)
        classifier = get_conv_classifier(num_classes, activation_fn, input_shape)
    elif option == "recurrent":
        print(X_train.shape, X_test.shape)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        # X_validation = X_validation.reshape(
        #     (X_validation.shape[0], X_validation.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
        print(X_train.shape, X_test.shape, "$$$$")
        input_shape = (X_train[0].shape)
        classifier = get_recurrent_classifier(num_classes, activation_fn, input_shape)
    elif option == "fully_connected":
        input_shape = (X_train[0].shape)
        classifier = get_fully_connected_classifier(num_classes, activation_fn, input_shape)
    else:
        raise Exception("Wrong option!")

    adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
    sgd_opt = tf.keras.optimizers.SGD(lr=learning_rate)
    opt = adam_opt

    # loss_function = tf.keras.losses.CategoricalCrossentropy()
    loss_function = tf.keras.losses.BinaryCrossentropy()

    # metrics_function = 'accuracy'
    metrics_function = tfa.metrics.F1Score(num_classes=2)

    classifier.compile(optimizer=opt, loss=loss_function, metrics=[metrics_function])

    validation_option = [None, "split", "data"][1]
    validation_split = 0.1  # 0.1 -> 1.5k out of 15k OR 0.07 -> 1k out of 15k
    # (we basically win 1500 new datapoints or 2000 new datapoints or alltogether 3000 datapoints)
    validation_data = (X_test, y_test)

    if validation_option == "split":
        X_train = np.concatenate((X_train, X_test))
        y_train = np.concatenate((y_train, y_test))

    print(X_train.shape, y_train.shape)

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


if __name__ == '__main__':
    pass
    # main()
