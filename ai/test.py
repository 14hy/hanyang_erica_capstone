import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import cv2

from torchvision import transforms

sys.path.append("../")

from ai.prepare_data import rnn_data2, rnn_data, image_loader_trash

TRASH_DATA_PATH = "data/trash1/train"
TRASH_DATA_PATH2 = "data/trash2/train"
DETECTOR_DATA_PATH = "data/detector/train"
BATCH_SIZE = 32
CKPT = "ckpts/classifier4.pth"


def test_image_data():
#     train_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "train")
    train_loader = image_loader_trash(BATCH_SIZE, True)

    x, y = next(iter(train_loader))

    for i in range(32):

        print(y[i])
        img = x[i].numpy()
        img = np.transpose(img, axes=(1, 2, 0)) + 0.5
        print(img.max(), img.min())
        plt.imshow(img)
        plt.show()


def test_rnn_data():
#     train_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE)
    train_loader = rnn_data2(BATCH_SIZE, True)

    x, y = next(train_loader)

    for i in range(8):
        img = x[7][i].numpy()
        img = np.transpose(img, axes=(1, 2, 0))
        print(img.max(), img.min())
        plt.imshow(img)
        plt.show()

        print(y[7])


def test_classifier():
    loader = rnn_data2(TRASH_DATA_PATH2, BATCH_SIZE)

    from ai_torch_ver.Classifier import Classifier
    clf = Classifier(4, 0.5).cuda()
    clf.load(CKPT)

    with torch.no_grad():
        test_score = 0.0
        cnt = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda().long()
            
            logps = clf(x_batch)
            score = clf.score(logps, y_batch)
            print(clf.predict(x_batch))
            test_score += score
            cnt += 1

            preds = clf.predict(x_batch)
            print(preds)

        test_score /= cnt

        print(f"Test score: {test_score:.6f}")


def test_detector_image():
    from prepare_data import get_detector_data

    loader = iter(get_detector_data(DETECTOR_DATA_PATH, batch_size=32))

    images = next(loader).numpy()

    for i in range(10):
        src = images[i, 0]
        pos = images[i, 1]
        neg = images[i, 2]

        plt.imshow(src.transpose(1, 2, 0))
        plt.show()

        plt.imshow(pos.transpose(1, 2, 0))
        plt.show()

        plt.imshow(neg.transpose(1, 2, 0))
        plt.show()


def test_siamese_image():
    from prepare_data import siamese_data_loader

    loader = iter(siamese_data_loader(TRASH_DATA_PATH, batch_size=16))

    x_src_batch, y_src_batch, x_pos_batch, y_pos_batch, x_neg_batch, y_neg_batch = next(loader)

    for i in range(10):
        src = x_src_batch[i].numpy()
        pos = x_pos_batch[i].numpy()
        neg = x_neg_batch[i].numpy()

        print(y_src_batch[i])
        print(y_pos_batch[i])
        print(y_neg_batch[i])

        plt.imshow(src.transpose(1, 2, 0))
        plt.show()

        plt.imshow(pos.transpose(1, 2, 0))
        plt.show()

        plt.imshow(neg.transpose(1, 2, 0))
        plt.show()


def test_encoder():
    from prepare_data import image_loader
    from features.StackedEncoder import StackedEncoder

    enc = StackedEncoder([False] * 3)
    enc.load("ckpts/encoder.pth")

    loader = iter(image_loader(TRASH_DATA_PATH, 16, True))

    images, labels = next(loader)

    encoded, decoded = enc(images, 0)
    for i in range(16):
        img_np = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.show()

        decoded_np = decoded[i].permute(1, 2, 0).numpy()
        plt.imshow(decoded_np)
        plt.show()


test_image_data()
# test_siamese_image()
# test_rnn_data()
# test_classifier()
# test_encoder()
