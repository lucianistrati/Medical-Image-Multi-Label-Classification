from src.solt_augmentation import augment_image
from src.train_neural_net import train_nn
from matplotlib import pyplot as plt
from collections import Counter
from typing import List

import pandas as pd
import numpy as np

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
    for i in range(len(df)):
        df.at[i, "label1"] = labels_1[i]
        df.at[i, "label2"] = labels_2[i]
        df.at[i, "label3"] = labels_3[i]
    df.to_csv("good_submission.csv", index=False)


def load_images_from_folder(folder_path: str, augment_images: bool = False):
    images = []
    num_channels = 1
    if augment_images:
        train_labels_df = pd.read_csv("data/train_labels.csv")
        val_labels_df = pd.read_csv("data/val_labels.csv")

    for file in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, file)
        image = cv2.imread(filepath)
        one_channel_image = np.reshape(image[:, :, 0], (64, 64, 1))
        if num_channels == 3:
            one_channel_image = cv2.cvtColor(one_channel_image, cv2.COLOR_GRAY2RGB)
        images.append(one_channel_image)
        if augment_images:
            if "test" in filepath:
                continue
            augmented_images = augment_image(one_channel_image)
            indexes = [str(i) for i in range(1, len(augmented_images) + 1)]
            paths = [filepath[:filepath.rfind(".png")] + "_" + index + ".png" for index in indexes]
            for (augmented_path, augmented_image) in zip(paths, augmented_images):
                cv2.imwrite(augmented_path, augmented_image)
                images.append(augmented_image)
                if "train" in filepath:
                    for j in range(len(train_labels_df)):
                        if train_labels_df.at[j, "id"] in filepath:
                            label_1 = train_labels_df.at[j, "label1"]
                            label_2 = train_labels_df.at[j, "label2"]
                            label_3 = train_labels_df.at[j, "label3"]
                if "val" in filepath:
                    for j in range(len(val_labels_df)):
                        if val_labels_df.at[j, "id"] in filepath:
                            label_1 = val_labels_df.at[j, "label1"]
                            label_2 = val_labels_df.at[j, "label2"]
                            label_3 = val_labels_df.at[j, "label3"]
                new_row = {"id": augmented_path[augmented_path.rfind("/") + 1:],
                            "label1": label_1,
                            "label2": label_2,
                            "label3": label_3}
                if "train" in filepath:
                    train_labels_df_copy = train_labels_df.append(new_row, ignore_index=True)
                    train_labels_df = train_labels_df_copy
                if "val" in filepath:
                    val_labels_df_copy = val_labels_df.append(new_row, ignore_index=True)
                    val_labels_df = val_labels_df_copy
    if augment_images:
        train_labels_df.to_csv("data/train_labels.csv")
        val_labels_df.to_csv("data/val_labels.csv")

    return images


def load_data(resize_to_32px: bool = False, convert_to_float32: bool = False, normalize: bool = False):
    train_images = load_images_from_folder("data/train_images")
    val_images = load_images_from_folder("data/val_images")
    test_images = load_images_from_folder("data/test_images")

    if resize_to_32px:
        train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
        val_images = val_images.reshape(val_images.shape[0], 32, 32, 3)
        test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

    if convert_to_float32:
        train_images = train_images.astype('float32')
        val_images = val_images.astype('float32')
        test_images = test_images.astype('float32')

    if normalize:
        # normalizing the data to help with the training
        train_images /= 255
        val_images /= 255
        test_images /= 255

    return train_images, val_images, test_images


def load_labels():
    train_labels = pd.read_csv("data/train_labels.csv")
    val_labels = pd.read_csv("data/val_labels.csv")
    return train_labels, val_labels

def plot_average_image(all_images, all_labels):
    neg_images = [image for (image, label) in zip(all_images, all_labels) if label == 0]
    pos_images = [image for (image, label) in zip(all_images, all_labels) if label == 1]

    avg_neg_image = np.mean(np.array(neg_images))
    avg_pos_image = np.mean(np.array(pos_images))

    plt.imshow(avg_pos_image, cmap='gray')
    plt.title("average with label 1 image")
    plt.show()

    plt.imshow(avg_neg_image, cmap='gray')
    plt.title("average with label 0 image")
    plt.show()

    difference_images = avg_pos_image - avg_neg_image

    plt.imshow(difference_images, cmap='gray')
    plt.title("difference between average label 1 image and average label 0 image")
    plt.show()

    print("Difference between images value:", np.mean(difference_images))


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

    return images, labels_1, labels_2, labels_3


def get_class_weight(labels):
    cnt = Counter(labels)
    return {0: cnt[0], 1: cnt[1]}


def main():
    train_images, val_images, test_images = load_data()

    train_labels, val_labels = load_labels()

    train_labels_1, train_labels_2, train_labels_3 = train_labels["label1"].to_list(), \
        train_labels["label2"].to_list(), \
        train_labels["label3"].to_list()

    val_labels_1, val_labels_2, val_labels_3 = val_labels["label1"].to_list(), \
        val_labels["label2"].to_list(), \
        val_labels["label3"].to_list()

    # train_images, train_labels_1, train_labels_2, train_labels_3 = keep_one_black_image(train_images, train_labels_1, train_labels_2, train_labels_3)
    # val_images, val_labels_1, val_labels_2, val_labels_3 = keep_one_black_image(val_images, val_labels_1, val_labels_2, val_labels_3)

    num_classes = 2
    model_name = "vanilla"

    X_train = np.array(train_images)
    X_test = np.array(val_images)
    X_submission = np.array(test_images)

    y_train = np.array(train_labels_1)
    y_test = np.array(val_labels_1)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    class_weight = get_class_weight(train_labels_1)
    labels_1, logging_metrics_list_1 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)

    y_train = np.array(train_labels_2)
    y_test = np.array(val_labels_2)
    class_weight = get_class_weight(train_labels_3)
    labels_2, logging_metrics_list_2 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)

    y_train = np.array(train_labels_3)
    y_test = np.array(val_labels_3)
    class_weight = get_class_weight(train_labels_3)
    labels_3, logging_metrics_list_3 = train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, X_submission, class_weight)

    create_sample_submission(labels_1, labels_2, labels_3)

    if logging_metrics_list_1 is not None and logging_metrics_list_2 is not None and logging_metrics_list_3 is not None:
        res = (float(logging_metrics_list_1[0][1]) + float(logging_metrics_list_3[0][1]) +
               float(logging_metrics_list_3[0][1])) / 3
        print("F1" * 20)
        print("Final F1:", res)
        print("F1" * 20)


if __name__ == '__main__':
    main()
