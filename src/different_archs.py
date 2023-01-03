from transformers import BeitFeatureExtractor
from transformers import ViTFeatureExtractor
from transformers import AutoFeatureExtractor
from tqdm import tqdm

import numpy as np

import cv2
import os


def swin_tiny_patch_window(image=None, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    inputs = feature_extractor(images=image, return_tensors="pt")

    # print(inputs["pixel_values"].shape)

    return inputs["pixel_values"]


def resnet_50(image=None, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")

    inputs = feature_extractor(image, return_tensors="pt")

    # print(inputs["pixel_values"].shape)

    return inputs["pixel_values"]


def vit_base_patch_16_384(image=None, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    inputs = feature_extractor(images=image, return_tensors="pt")
    # print(inputs["pixel_values"].shape)

    return inputs["pixel_values"]


def vit_base_patch_16_224(image=None, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    inputs = feature_extractor(images=image, return_tensors="pt")

    # print(inputs["pixel_values"].shape)

    return inputs["pixel_values"]


def beit_base(image=None, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    inputs = feature_extractor(images=image, return_tensors="pt")

    # print(inputs["pixel_values"].shape)

    return inputs["pixel_values"]


def vit_base_32_224_21k(image=None, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-224-in21k')

    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # print(inputs["pixel_values"].shape)

    return inputs["pixel_values"]


def vit_base_16_224_21k(image=None, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    inputs = feature_extractor(images=image, return_tensors="pt")

    # print(inputs["pixel_values"].shape)

    return inputs["pixel_values"]


def load_feature_extractor(name: str):
    if name == "google/vit-base-patch16-224-in21k":
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    elif name == "google/vit-base-patch32-224-in21k":
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-224-in21k')
    elif name == "google/vit-base-patch16-224":
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    elif name == "google/vit-base-patch16-384":
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    elif name == "microsoft/swin-tiny-patch4-window7-224":
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    elif name == "microsoft/resnet-50":
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
    elif name == "microsoft/beit-base-patch16-224-pt22k-ft22k":
        feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    else:
        raise Exception(f"Wrong name: {name}!")
    return feature_extractor


def inference(image, feature_extractor, name):
    if name == "google/vit-base-patch16-224-in21k":
        features = vit_base_16_224_21k(image, feature_extractor)
    elif name == "google/vit-base-patch32-224-in21k":
        features = vit_base_32_224_21k(image, feature_extractor)
    elif name == "google/vit-base-patch16-224":
        features = vit_base_patch_16_224(image, feature_extractor)
    elif name == "google/vit-base-patch16-384":
        features = vit_base_patch_16_384(image, feature_extractor)
    elif name == "microsoft/swin-tiny-patch4-window7-224":
        features = swin_tiny_patch_window(image, feature_extractor)
    elif name == "microsoft/resnet-50":
        features = resnet_50(image, feature_extractor)
    elif name == "microsoft/beit-base-patch16-224-pt22k-ft22k":
        features = beit_base(image, feature_extractor)
    else:
        raise Exception(f"Wrong name: {name}!")
    return features


def create_features_set(images, names, type_of_data):
    print(type_of_data, len(images))
    for name in tqdm(names):
        print(name)
        if os.path.exists(f"data/{type_of_data}_with_{name}_features.npy"):
            continue
        # if name == "google/vit-base-patch16-224-in21k" or name == "google/vit-base-patch32-224-in21k"\
        #         or name == "microsoft/beit-base-patch16-224-pt22k-ft22k":
        #     continue
        feature_extractor = load_feature_extractor(name)
        features_list = []
        for image in tqdm(images):
            features = inference(image, feature_extractor, name)
            features_list.append(features)
        features_list = np.array(features_list)
        name = name.replace("/", "_")
        np.save(arr=features_list, file=f"data/{type_of_data}_with_{name}_features.npy")


def main():
    image = cv2.imread("data/train_images/10004.png")
    print(image.shape)

    feature_extractor = None
    load_feature_extractor = True

    if load_feature_extractor:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    features = vit_base_16_224_21k(image, feature_extractor)
    print(features.shape)

    if load_feature_extractor:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-224-in21k')
    features = vit_base_32_224_21k(image)
    print(features.shape, feature_extractor)

    if load_feature_extractor:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    features = vit_base_patch_16_224(image)
    print(features.shape, feature_extractor)

    if load_feature_extractor:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    features = vit_base_patch_16_384(image)
    print(features.shape, feature_extractor)

    if load_feature_extractor:
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    features = swin_tiny_patch_window(image, feature_extractor)
    print(features.shape, feature_extractor)

    if load_feature_extractor:
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
    features = resnet_50(image, feature_extractor)
    print(features.shape)

    if load_feature_extractor:
        feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    features = beit_base(image, feature_extractor)
    print(features.shape)

    names = ["google/vit-base-patch16-224-in21k", "google/vit-base-patch32-224-in21k",
             "microsoft/beit-base-patch16-224-pt22k-ft22k", "google/vit-base-patch16-224",
             "google/vit-base-patch16-384", "microsoft/resnet-50", "microsoft/swin-tiny-patch4-window7-224"]
    print(names)
    images = []
    from src.main import load_data

    train_images, val_images, test_images = load_data()

    create_features_set(train_images, names, "train")
    create_features_set(val_images, names, "val")
    create_features_set(test_images, names, "test")

    # https://colab.research.google.com/github/nateraw/huggingpics/blob/main/HuggingPics.ipynb


if __name__ == "__main__":
    main()
