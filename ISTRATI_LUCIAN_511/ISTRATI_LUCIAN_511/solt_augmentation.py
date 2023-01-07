from solt.transforms import Rotate, Flip, Shear, Scale, Pad, Blur, Crop, CutOut

import matplotlib.pyplot as plt

import random
import solt
import cv2


def augment_image(input_img, visualize: bool = False):
    # https://www.analyticsvidhya.com/blog/2022/04/image-augmentation-using-3-python-libraries/
    h, w, c = input_img.shape
    img = input_img[:w]
    # creating the stream of augmentations
    stream = solt.Stream([
        Rotate(angle_range=(-90, 90), p=1, padding='r'),  # rotations between -90 and 90 degrees
        Flip(axis=1, p=0.5),  # horizontal flip
        Flip(axis=0, p=0.5),  # vertical flip
        Shear(range_x=0.3, range_y=0.8, p=0.5, padding='r'),
        Scale(range_x=(0.8, 1.3), padding='r', range_y=(0.8, 1.3), same=False, p=0.5),  # rescale
        Pad((w, h), 'r'),  # paddinngs
        Crop((w, w), 'r'),  # croppings
        Blur(k_size=7, blur_type='m'),  # blurrings
        solt.SelectiveStream([
            CutOut(40, p=1),
            CutOut(50, p=1),
            CutOut(10, p=1),
            solt.Stream(),
            solt.Stream(),
        ], n=3),
    ], ignore_fast_mode=True)
    fig = plt.figure(figsize=(17, 17))
    n_augs = 10
    random.seed(2)
    augmented_images = []
    for i in range(n_augs):
        img_aug = stream({'image': img}, return_torch=False, ).data[0].squeeze()
        augmented_images.append(img_aug)
        if visualize:
            ax = fig.add_subplot(1, n_augs, i + 1)
            if i == 0:
                ax.imshow(img)
            else:
                ax.imshow(img_aug)
            ax.set_xticks([])
            ax.set_yticks([])
    if visualize:
        plt.show()

    return augmented_images


def main():
    img = cv2.imread("data/val_images/40000.png")

    augmented_images = augment_image(img)

    print(len(augmented_images))
    print(augmented_images[0].shape)


if __name__ == "__main__":
    main()
