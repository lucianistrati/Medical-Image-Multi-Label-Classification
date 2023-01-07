from src.main import load_data, load_labels
import matplotlib.pyplot as plt

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def main():
    train_images, val_images, test_images = load_data()

    train_labels, val_labels = load_labels()
    train_val_labels = train_labels + val_labels

    train_labels_1, train_labels_2, train_labels_3 = train_labels["label1"].to_list(), \
        train_labels["label2"].to_list(), \
        train_labels["label3"].to_list()

    val_labels_1, val_labels_2, val_labels_3 = val_labels["label1"].to_list(), \
        val_labels["label2"].to_list(), \
        val_labels["label3"].to_list()

    for cls in [0, 1, 2]:
        for type_of_set in ["validation", "training"]:
            objects = ('0 label', '1 label')
            y_pos = np.arange(len(objects))

            if type_of_set == "validation":
                if cls == 0:
                    cnt = Counter(val_labels_1)
                if cls == 1:
                    cnt = Counter(val_labels_2)
                if cls == 2:
                    cnt = Counter(val_labels_3)

            if type_of_set == "training":
                if cls == 0:
                    cnt = Counter(train_labels_1)
                if cls == 1:
                    cnt = Counter(train_labels_2)
                if cls == 2:
                    cnt = Counter(train_labels_3)

            performance = [cnt[0], cnt[1]]

            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('Labels')
            plt.title(f'Labels distribution for class {cls} in {type_of_set} set')
            plt.show()
            plt.savefig(f'data/LabelsDistributionForClass{cls}in{type_of_set}set.png')


if __name__ == "__main__":
    main()
