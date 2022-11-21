from typing import List
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import random
import cv2
import os


def create_sample_submission(labels_1: List = None, labels_2: List = None, labels_3: List = None):
    if labels_1 is None:
        labels_1 = np.load(allow_pickle=True, file=f"data/y_pred_for_submission_1.npy")
    if labels_2 is None:
        labels_2 = np.load(allow_pickle=True, file=f"data/y_pred_for_submission_2.npy")
    if labels_3 is None:
        labels_3 = np.load(allow_pickle=True, file=f"data/y_pred_for_submission_3.npy")
    df = pd.read_csv("data/sample_submission.csv")
    # id,label1,label2,label3
    for i in range(len(df)):
        df.at[i, "label1"] = labels_1[i]
        df.at[i, "label2"] = labels_2[i]
        df.at[i, "label3"] = labels_3[i]
    df.to_csv("good_submission.csv", index=False)


def load_images_from_folder(folder_path: str):
    images = []
    for file in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, file)
        image = cv2.imread(filepath)
        images.append(image)
    return images


def load_data():
    train_images = load_images_from_folder("data/train_images")
    val_images = load_images_from_folder("data/val_images")
    test_images = load_images_from_folder("data/test_images")
    return train_images, val_images, test_images


def load_labels():
    train_labels = pd.read_csv("data/train_labels.csv")
    val_labels = pd.read_csv("data/val_labels.csv")
    return train_labels, val_labels


def train_model(X_train, y_train, X_val, y_val, X_test, label):
    print(len(X_train), len(y_train), len(X_val), len(y_val))
    model = SVC(class_weight="balanced")
    print("PRE FIT")
    model.fit(X_train, y_train)
    del X_train
    del y_train
    print("POST FIT")
    y_pred = model.predict(X_val)
    del X_val
    print(f1_score(y_pred, y_val))
    del y_pred
    del y_val
    y_pred_for_submission = model.predict(X_test)
    del X_test
    del model
    np.save(allow_pickle=True, arr=y_pred_for_submission, file=f"data/y_pred_for_submission_{label}.npy")
    return y_pred_for_submission


def plot_average_image(all_images, all_labels):
    neg_images = [image for (image, label) in zip(all_images, all_labels) if label == 0]
    pos_images = [image for (image, label) in zip(all_images, all_labels) if label == 1]

    avg_neg_image = np.mean(np.array(neg_images))
    avg_pos_image = np.mean(np.array(pos_images))

    plt.imshow(avg_pos_image, cmap='gray')  # , vmin=0, vmax=255)
    plt.title("average with label 1 image")
    plt.show()

    plt.imshow(avg_neg_image, cmap='gray')  # , vmin=0, vmax=255)
    plt.title("average with label 0 image")
    plt.show()


from src.train_neural_net import train_nn


# TODO cut second and third dimensions, keep just the first one since they are identical, so that will
# be a (64, 64, 1) instead of (64, 64, 3)
def main():
    train_images, val_images, test_images = load_data()
    train_val_images = train_images + val_images
    train_labels, val_labels = load_labels()

    train_labels_1, train_labels_2, train_labels_3 = train_labels["label1"].to_list(), \
                                                     train_labels["label2"].to_list(), \
                                                     train_labels["label3"].to_list()

    val_labels_1, val_labels_2, val_labels_3 = val_labels["label1"].to_list(), \
                                               val_labels["label2"].to_list(), \
                                               val_labels["label3"].to_list()

    from collections import Counter

    def get_class_weight(labels):
        cnt = Counter(labels)
        return {0: cnt[0], 1: cnt[1]}

    X_train = np.array(train_images)
    X_test = np.array(val_images)
    X_submission = np.array(test_images)
    num_classes = 2
    model_name = "vanilla"

    y_train = np.array(train_labels_1)
    y_test = np.array(val_labels_1)
    class_weight = get_class_weight(train_labels_1)
    labels_1, logging_metrics_list_1 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)

    np.save(allow_pickle=True, arr=labels_1, file=f"data/y_pred_for_submission_11111.npy")

    y_train = np.array(train_labels_2)
    y_test = np.array(val_labels_2)
    class_weight = get_class_weight(train_labels_3)
    labels_2, logging_metrics_list_2 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)

    np.save(allow_pickle=True, arr=labels_2, file=f"data/y_pred_for_submission_22222.npy")

    y_train = np.array(train_labels_3)
    y_test = np.array(val_labels_3)
    class_weight = get_class_weight(train_labels_3)
    labels_3, logging_metrics_list_3 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)

    np.save(allow_pickle=True, arr=labels_3, file=f"data/y_pred_for_submission_33333.npy")

    create_sample_submission(labels_1, labels_2, labels_3)

    if logging_metrics_list_1 is not None and logging_metrics_list_2 is not None and logging_metrics_list_3 is not None:
        res = (float(logging_metrics_list_1[0][1]) + float(logging_metrics_list_3[0][1]) +
               float(logging_metrics_list_3[0][1])) / 3
        print("Final F1:", res)

    # train_images = [train_image.flatten() for train_image in train_images]
    # val_images = [val_image.flatten() for val_image in val_images]
    # test_images = [test_image.flatten() for test_image in test_images]

    # labels_1 = train_model(X_train=train_images, y_train=train_labels_1, X_val=val_images, y_val=val_labels_1,
    #                        X_test=test_images, label="12")
    # labels_2 = train_model(X_train=train_images, y_train=train_labels_2, X_val=val_images, y_val=val_labels_2,
    #                        X_test=test_images, label="22")
    # labels_3 = train_model(X_train=train_images, y_train=train_labels_3, X_val=val_images, y_val=val_labels_3,
    #                        X_test=test_images, label="32")


if __name__ == '__main__':
    main()
