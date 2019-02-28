import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib as path
import glob
import cv2

sys.path.append(".")

from StackedEncoder import StackedEncoder
# from FeatureCNN import FeatureCNN
from Classifier import Classifier
from keras.utils import to_categorical


def load_dataset():
    from keras.datasets.cifar10 import load_data

    (X_train, Y_train), (X_test, Y_test) = load_data()

    X_train_64 = np.zeros((50000, 48, 48, 3))
    X_test_64 = np.zeros((10000, 48, 48, 3))
    Y_train_64 = np.zeros((50000, 10))
    Y_test_64 = np.zeros((10000, 10))

    for i in range(50000):
        X_train_64[i] = cv2.resize(X_train[i], dsize=(48, 48))
        Y_train_64[i, Y_train[i]] = 1
    for i in range(10000):
        X_test_64[i] = cv2.resize(X_test[i], dsize=(48, 48))
        Y_test_64[i, Y_test[i]] = 1

    X_train_64 /= 255
    X_test_64 /= 255

    print("Data Information:")
    print(X_train_64.shape, X_test_64.shape)
    print(np.min(X_train_64), np.max(X_train_64))
    print()

    return X_train_64, Y_train_64, X_test_64, Y_test_64


def load_food_dataset_with_label(num_classes=101):
    data_path = "/home/jylee/Dataset/food-101"
    images_path = data_path + "/images"
    meta_path = data_path + "/meta"

    print("Loading data...")

    n_tr = num_classes * 750
    n_ts = num_classes * 250

    images_train = np.zeros((n_tr, 48, 48, 3))
    labels_train = np.zeros((n_tr, num_classes))

    images_test = np.zeros((n_ts, 48, 48, 3))
    labels_test = np.zeros((n_ts, num_classes))

    labels_dict = dict()
    label_index = 0

    with open(meta_path + "/classes.txt") as f:
        for i, line in enumerate(f):
            if i == num_classes:
                break
            line = line.replace("\n", "").replace("\r", "")

            labels_dict[line] = label_index
            label_index += 1

    print("Loading train set...")

    with open(meta_path + "/train.txt") as f:
        index = 0

        for line in f:
            lst = line.split(sep="/")
            if lst[0] not in labels_dict:
                continue

            labels_train[index, labels_dict[lst[0]]] = 1

            img_path = images_path + "/" + line.replace("\n", "").replace("\r", "") + ".jpg"
            img = plt.imread(img_path).astype(np.float32) / 255
            img_resized = cv2.resize(img, dsize=(48, 48))

            if len(img.shape) == 3:
                images_train[index] = img_resized
            else:
                images_train[index, :, :, 0] = img_resized
                images_train[index, :, :, 1] = img_resized
                images_train[index, :, :, 2] = img_resized

            index += 1

            if index % 1000 == 0:
                print("Loaded train data: {}".format(index))

    print("Train set {} were loaded.".format(index))

    print("Loading test set...")

    with open(meta_path + "/test.txt") as f:
        index = 0

        for line in f:
            lst = line.split(sep="/")
            if lst[0] not in labels_dict:
                continue

            labels_test[index, labels_dict[lst[0]]] = 1

            img_path = images_path + "/" + line.replace("\n", "").replace("\r", "") + ".jpg"
            img = plt.imread(img_path).astype(np.float32) / 255
            img_resized = cv2.resize(img, dsize=(48, 48))

            if len(img.shape) == 3:
                images_test[index] = img_resized
            else:
                images_test[index, :, :, 0] = img_resized
                images_test[index, :, :, 1] = img_resized
                images_test[index, :, :, 2] = img_resized

            index += 1

            if index % 1000 == 0:
                print("Loaded test data: {}".format(index))

    print("Test set () were loaded.".format(index))

    return images_train, labels_train, images_test, labels_test


def load_food_dataset(num_classes=101):
    data_path = "/home/jylee/datasets/food-101"
    images_path = data_path + "/images"

    print("Loading data...")

    images = np.zeros((num_classes * 1000, 48, 48, 3))

    idx = 0
    for d in path.Path(images_path).iterdir():
        print("Directory {} in".format(d))
        for f in d.glob("*.*"):
            img = plt.imread(f).astype(np.float32) / 255

            if len(img.shape) == 2:
                images[idx, :, :, 0] = cv2.resize(img, dsize=(48, 48))
                images[idx, :, :, 1] = images[idx, :, :, 0]
                images[idx, :, :, 2] = images[idx, :, :, 0]
            else:
                images[idx, :, :, :] = cv2.resize(img, dsize=(48, 48))

            idx += 1

        print("Directory {} out".format(d))
        print("Processed data: {}".format(idx))

        if idx == images.shape[0]:
            break

    print("Data info:")
    print(images.shape)

    print("Loading completed.")

    return images


def load_fruits_dataset():
    print("Loading data...")

    data_path = "D:/Dataset/fruits-360"
    images_path = data_path + "/Training"

    num_file = len(glob.glob(images_path + "/*/*"))
    num_classes = len(path.Path(images_path).dirs())

    images_train = np.zeros((num_file, 48, 48, 3))
    labels_train = np.zeros((num_file, num_classes))

    label_dict = dict()

    idx = 0
    count = 0

    for p in path.Path(images_path).dirs():
        label_dict[path.Path(p).basename()] = idx

        for f in glob.glob(p + "/*.jp*g"):
            image = plt.imread(f).astype(np.float32) / 255
            images_train[count, :, :, :] = cv2.resize(image, dsize=(48, 48))
            labels_train[count, idx] = 1

            count += 1

        idx += 1

        print("Loaded data: {}".format(count))

    print("Data loaded.")

    return images_train, labels_train

def load_trash_data():
    data_path = "/home/jylee/datasets/trashdata"

    print("Loading data...")

    images = np.zeros((2527, 48, 48, 3))

    index = 0
    for d in path.Path(data_path).iterdir():
        print(d)
        for f in d.glob("*.jp*g"):
            img = plt.imread(f).astype(np.float32) / 255
            img = cv2.resize(img, dsize=(48, 48))

            images[index] = img
            index += 1

    print("{} Data were loaded.".format(index))

    return images

def rnn_data_refine(X, Y, data_size, num_step, num_classes):
    m, h, w, c = np.shape(X)

    X_refined = np.zeros((data_size, num_step, h, w, c))
    Y_refined = np.zeros((data_size, num_classes))

    for clz in range(num_classes):
        samples = X[np.argmax(Y, axis=1) == clz, :, :, :]
        r = np.random.choice(np.arange(len(samples)), size=((data_size // num_classes) * num_step,))
        samples2 = samples[r]

        X_refined[(data_size // num_classes) * clz: (data_size // num_classes) * (clz + 1), :, :, :,
        :] = samples2.reshape((data_size // num_classes), num_step, h, w, c)
        Y_refined[(data_size // num_classes) * clz: (data_size // num_classes) * (clz + 1), clz] = 1

    return X_refined, Y_refined


def train_vaild_split(X_data, Y_data, valid_size):
    assert (X_data.shape[0] == Y_data.shape[0])

    num_classes = Y_data.shape[1]
    n = X_data.shape[0]
    shape = X_data.shape[1:]

    total = np.zeros((num_classes,))
    cnt = np.zeros((num_classes,))
    for i in range(num_classes):
        total[i] = int(X_data[np.argmax(Y_data, axis=1).reshape(-1) == i].shape[0])
        cnt[i] = int(X_data[np.argmax(Y_data, axis=1).reshape(-1) == i].shape[0] * valid_size)

    X_train = np.zeros((n - int(cnt.sum()), *shape))
    X_valid = np.zeros((int(cnt.sum()), *shape))
    Y_train = np.zeros((n - int(cnt.sum()), num_classes))
    Y_valid = np.zeros((int(cnt.sum()), num_classes))

    index = 0
    index_val = 0
    for i in range(num_classes):
        r = np.arange(int(total[i]))
        np.random.shuffle(r)

        X_train[index: index + int(total[i] - cnt[i])] = X_data[np.argmax(Y_data, axis=1).reshape(-1) == i][
            r[int(cnt[i]):]]
        Y_train[index: index + int(total[i] - cnt[i])] = Y_data[np.argmax(Y_data, axis=1).reshape(-1) == i][
            r[int(cnt[i]):]]

        X_valid[index_val: index_val + int(cnt[i])] = X_data[np.argmax(Y_data, axis=1).reshape(-1) == i][
            r[: int(cnt[i])]]
        Y_valid[index_val: index_val + int(cnt[i])] = Y_data[np.argmax(Y_data, axis=1).reshape(-1) == i][
            r[: int(cnt[i])]]

    return X_train, Y_train, X_valid, Y_valid


def train_encoder():
    # images = load_food_dataset(num_classes=101)
    images = load_trash_data()

    # images = np.concatenate([images_train, images_test], axis=0)

    shapes = [(48, 48, 3), (24, 24, 8), (12, 12, 8)]
    encoded_shape = (6, 6, 8)
    stacked_encoder = StackedEncoder()
    stacked_encoder.build(shapes, "/gpu:0", eta=[1e-5, 5e-5, 1e-4])
    # stacked_encoder.load_weights()
    stacked_encoder.fit(images, [1, 2], overwrite=True, epochs=[100, 200, 300], batch_size=128)


def train_classifier():
    images_train, labels_train, images_test, labels_test = load_dataset()
    # img, lbl = load_fruits_dataset()

    # images_train, labels_train, images_test, labels_test = load_food_dataset_with_label(101)

    print(images_train.shape, labels_train.shape)
    print(images_test.shape, labels_test.shape)

    # plt.imshow(images_test[6000])
    # plt.show()
    # print(labels_train[6000])

    m, h, w, c = images_train.shape
    m, num_classes = labels_train.shape
    num_step = 10
    num_gpu = 1

    X_train, Y_train = rnn_data_refine(images_train, labels_train, 20000, num_step, num_classes)

    # print("Train:")
    # print(X_train.shape, Y_train.shape)

    shapes = [(48, 48, 3), (24, 24, 8), (12, 12, 8)]
    encoded_shape = (6, 6, 8)

    clf = Classifier(num_step, num_classes, num_gpu)
    clf.build(h, w, c, eta=1e-3, load_weights=False, batch_size=128)

    clf.fit(X_train, Y_train, epochs=50)

    # X_valid, Y_valid = rnn_data_refine(images_test, labels_test, 8000, num_step, num_classes)

    print(clf.score(X_train, Y_train))


# print(clf.score(X_valid, Y_valid))

def validate_classifier():
    train_img, train_lbl, test_img, test_lbl = load_dataset_lego()

    print(train_img.shape, train_lbl.shape)
    print(test_img.shape, test_lbl.shape)

    m, h, w, c = train_img.shape
    m, num_classes = train_lbl.shape
    num_step = 6
    num_gpu = 2

    X_valid, Y_valid = rnn_data_refine(test_img, test_lbl, 15000, num_step, num_classes)
    print("Valid:")
    print(X_valid.shape, Y_valid.shape)

    shapes = [(48, 48, 3), (24, 24, 8), (12, 12, 8)]
    encoded_shape = (6, 6, 8)

    clf = Classifier(num_step, num_classes, num_gpu)
    clf.build(h, w, c)

    clf.start()
    print(clf.score(X_valid, Y_valid))
    clf.close()


train_encoder()
# train_classifier()
# validate_classifier()