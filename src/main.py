from typing import List
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from collections import Counter
from tqdm import tqdm

import pandas as pd
import numpy as np

import random
import cv2
import os

import efficientnet.keras as efn
import torch

efficient_net_model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'


def efficient_net_fn(img=None):
    # models can be build with Keras or Tensorflow frameworks
    # use keras and tfkeras modules respectively
    # efficientnet.keras / efficientnet.tfkeras

    if img is None:
        img = cv2.imread("data/train_images/10000.png")
    resized_img = cv2.resize(img, dsize=(224, 224))
    # print(resized_img.shape)
    resized_img = np.reshape(resized_img, (1, 224, 224, 3))
    output = efficient_net_model(resized_img)
    # print(type(output))
    # print(output.shape)
    return output

    # model use some custom objects, so before loading saved model
    # import module your network was build with
    # e.g. import efficientnet.keras / import efficientnet.tfkeras
    # import efficientnet.tfkeras
    # from tensorflow.keras.models import load_model
    # model = load_model('path/to/model.h5')


# efficient_net()

# a = 1/0

# OBSERVATION ------------ besides the 5000 black images, there is no other duplicate in the dataset
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
        one_channel_image = np.reshape(image[:, :, 0], (64, 64, 1))
        one_channel_image = cv2.cvtColor(one_channel_image, cv2.COLOR_GRAY2RGB)

        # print(one_channel_image.shape)
        # a = 1/0
        images.append(one_channel_image)
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

    difference_images = avg_pos_image - avg_neg_image

    plt.imshow(difference_images, cmap='gray')  # , vmin=0, vmax=255)
    plt.title("difference between average label 1 image and average label 0 image")
    plt.show()

    print("Difference between images value:", np.mean(difference_images))



from src.train_neural_net import train_nn


# TODO cut second and third dimensions, keep just the first one since they are identical, so that will
# be a (64, 64, 1) instead of (64, 64, 3)
def main():
    train_images, val_images, test_images = load_data()

    train_val_images = train_images + val_images

    # train_images = [tuple(img.flatten().tolist()) for img in train_images if img.sum() !=0]
    # val_images = [tuple(img.flatten().tolist()) for img in val_images if img.sum() != 0]
    # test_images = [tuple(img.flatten().tolist()) for img in test_images if img.sum() != 0]
    #
    # print(len(train_images), len(set(train_images)))
    # print(len(val_images), len(set(val_images)))
    # print(len(test_images), len(set(test_images)))
    # a = 1/0

    train_labels, val_labels = load_labels()
    train_val_labels = train_labels + val_labels

    train_labels_1, train_labels_2, train_labels_3 = train_labels["label1"].to_list(), \
                                                     train_labels["label2"].to_list(), \
                                                     train_labels["label3"].to_list()

    val_labels_1, val_labels_2, val_labels_3 = val_labels["label1"].to_list(), \
                                               val_labels["label2"].to_list(), \
                                               val_labels["label3"].to_list()

    def keep_one_black_image(images, labels_1, labels_2, labels_3):
        first_time = True
        to_remove = []
        first_idx = None
        for i, (img, label) in enumerate(zip(images, labels_1)):
            img_sum = img.sum()
            if first_time is False and img_sum == 0:
                to_remove.append(i)
                images.pop(i)
                labels_1.pop(i)
                labels_2.pop(i)
                labels_3.pop(i)
            if img_sum == 0:
                first_idx = i
                first_time = False
        labels_1[first_idx] = 0
        labels_2[first_idx] = 0
        labels_3[first_idx] = 0
        # for idx in to_remove:
        #     images.pop(idx)
        #     labels.pop(idx)
        return images, labels_1, labels_2, labels_3

    train_images, train_labels_1, train_labels_2, train_labels_3 = keep_one_black_image(train_images, train_labels_1, train_labels_2, train_labels_3)

    # train_images, train_labels_2 = keep_one_black_image(train_images, train_labels_2)
    # train_images, train_labels_3 = keep_one_black_image(train_images, train_labels_3)
    # train_images, train_labels_2 = keep_one_black_image(train_images, train_labels_2)
    # train_images, train_labels_1 = keep_one_black_image(train_images, train_labels_1)

    val_images, val_labels_1, val_labels_2, val_labels_3 = keep_one_black_image(val_images, val_labels_1, val_labels_2, val_labels_3)

    # val_images, val_labels_1 = keep_one_black_image(val_images, val_labels_1)
    # val_images, val_labels_2 = keep_one_black_image(val_images, val_labels_2)
    # val_images, val_labels_3 = keep_one_black_image(val_images, val_labels_3)

    # train_val_labels_1, train_val_labels_2, train_val_labels_3 = train_labels_1 + val_labels_1, \
    #                                                              train_labels_2 + val_labels_2, \
    #                                                              train_labels_3 + val_labels_3

    # bad_labelled_1 = {0: 0, 1: 0}
    # bad_labelled_2 = {0: 0, 1: 0}
    # bad_labelled_3 = {0: 0, 1: 0}
    #
    # for (img, l_1, l_2, l_3) in zip(train_val_images, train_val_labels_1, train_val_labels_2, train_val_labels_3):
    #     if img.sum() == 0:
    #         # print(l_1, l_2, l_3)
    #         if l_1 == 0:
    #             bad_labelled_1[0] += 1
    #         elif l_1 == 1:
    #             bad_labelled_1[1] += 1
    #         else:
    #             raise Exception("1")
    #
    #         if l_2 == 0:
    #             bad_labelled_2[0] += 1
    #         elif l_2 == 1:
    #             bad_labelled_2[1] += 1
    #         else:
    #             raise Exception("2")
    #
    #         if l_3 == 0:
    #             bad_labelled_3[0] += 1
    #         elif l_3 == 1:
    #             bad_labelled_3[1] += 1
    #         else:
    #             raise Exception("3")
    #
    # print("Bad labelled black images:")
    # print("Feature no. 1: ", bad_labelled_1)
    # print("Feature no. 2: ", bad_labelled_2)
    # print("Feature no. 3: ", bad_labelled_3)

    """
    if black -> 0,0,0
    else: sample weights on training for non black images and 
    """
    # TOOD make a set out of all the black images and give them the 0 label
    # from tqdm import tqdm
    # in_train_and_val = 0
    # for img in tqdm(val_images):
    #     if img.sum() != 0:
    #         for img_ in test_images:
    #             # if img_.sum() != 0:
    #             if np.array_equal(img, img_):
    #                 in_train_and_val += 1
    #
    # print("Common images in train and val: ", in_train_and_val)
    # a = 1 / 0

    def get_class_weight(labels):
        cnt = Counter(labels)
        return {0: cnt[0], 1: cnt[1]}

    X_train = np.array(train_images)
    X_test = np.array(val_images)
    X_submission = np.array(test_images)
    num_classes = 2
    model_name = "vanilla"

    # y_train = np.array(train_labels_1)
    # y_test = np.array(val_labels_1)
    # class_weight = get_class_weight(train_labels_1)
    # labels_1, logging_metrics_list_1 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)
    #
    # np.save(allow_pickle=True, arr=labels_1, file=f"data/y_pred_for_submission_11111.npy")
    #
    # y_train = np.array(train_labels_2)
    # y_test = np.array(val_labels_2)
    # class_weight = get_class_weight(train_labels_3)
    # labels_2, logging_metrics_list_2 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)
    #
    # np.save(allow_pickle=True, arr=labels_2, file=f"data/y_pred_for_submission_22222.npy")
    #
    # y_train = np.array(train_labels_3)
    # y_test = np.array(val_labels_3)
    # class_weight = get_class_weight(train_labels_3)
    # labels_3, logging_metrics_list_3 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)
    #
    # np.save(allow_pickle=True, arr=labels_3, file=f"data/y_pred_for_submission_33333.npy")
    #
    # create_sample_submission(labels_1, labels_2, labels_3)
    #
    # if logging_metrics_list_1 is not None and logging_metrics_list_2 is not None and logging_metrics_list_3 is not None:
    #     res = (float(logging_metrics_list_1[0][1]) + float(logging_metrics_list_3[0][1]) +
    #            float(logging_metrics_list_3[0][1])) / 3
    #     print("Final F1:", res)

    # train_images = [efficient_net_fn(train_image) for train_image in tqdm(train_images)]
    # train_images = np.array(train_images)
    train_images = np.load(file="eff_net_train.npy", allow_pickle=True)
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[2])

    print(train_images.shape)
    # np.save(file="eff_net_train.npy", arr=train_images, allow_pickle=True)

    # val_images = [efficient_net_fn(val_image) for val_image in tqdm(val_images)]
    # val_images = np.array(val_images)
    val_images = np.load(file="eff_net_val.npy", allow_pickle=True)
    val_images = val_images.reshape(val_images.shape[0], val_images.shape[2])
    print(val_images.shape)

    # np.save(file="eff_net_val.npy", arr=val_images, allow_pickle=True)

    # test_images = [efficient_net_fn(test_image) for test_image in tqdm(test_images)]
    # test_images = np.array(test_images)
    test_images = np.load(file="eff_net_test.npy", allow_pickle=True)
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[2])

    print(test_images.shape)

    # a = 1/0

    # np.save(file="eff_net_test.npy", arr=test_images, allow_pickle=True)

    # train_images = [train_image.flatten() for train_image in train_images]
    # val_images = [val_image.flatten() for val_image in val_images]
    # test_images = [test_image.flatten() for test_image in test_images]

    labels_1 = train_model(X_train=train_images, y_train=train_labels_1, X_val=val_images, y_val=val_labels_1,
                           X_test=test_images, label="122_effnet")
    labels_2 = train_model(X_train=train_images, y_train=train_labels_2, X_val=val_images, y_val=val_labels_2,
                           X_test=test_images, label="222_effnet")
    labels_3 = train_model(X_train=train_images, y_train=train_labels_3, X_val=val_images, y_val=val_labels_3,
                           X_test=test_images, label="322_effnet")

    create_sample_submission(labels_1, labels_2, labels_3)

"""
TODO: ideas for neural nets:
https://towardsdatascience.com/train-a-neural-network-to-detect-breast-mri-tumors-with-pytorch-250a02be7777
https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418
https://ecode.dev/cnn-for-medical-imaging-using-tensorflow-2/
"""

if __name__ == '__main__':
    main()
