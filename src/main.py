from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import pandas as pd
import numpy as np

import cv2
import os


def create_sample_submission(labels_1, labels_2, labels_3):
    df = pd.read_csv("data/sample_submission.csv")
    # id,label1,label2,label3
    for i in range(len(df)):
        df.at[i, "label1"] = labels_1[i]
        df.at[i, "label2"] = labels_2[i]
        df.at[i, "label3"] = labels_3[i]
    df.to_csv("good_submission.csv")


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
    print("POST FIT")
    y_pred = model.predict(X_val)
    print(accuracy_score(y_pred, y_val))
    y_pred_for_submission = model.predict(X_test)
    np.save(allow_pickle=True, arr=y_pred_for_submission, file=f"data/y_pred_for_submission_{label}.npy")
    return y_pred


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

    train_images = [train_image.flatten() for train_image in train_images]
    val_images = [val_image.flatten() for val_image in val_images]

    labels_1 = train_model(train_images, train_labels_1, val_images, val_labels_1, test_images, label="1")
    labels_2 = train_model(train_images, train_labels_2, val_images, val_labels_2, test_images, label="2")
    labels_3 = train_model(train_images, train_labels_3, val_images, val_labels_3, test_images, label="3")

    create_sample_submission(labels_1, labels_2, labels_3)


if __name__ == '__main__':
    main()
