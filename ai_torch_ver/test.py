import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append("../")

from ai_torch_ver.prepare_data import rnn_data

TRASH_DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/total"
DETECTOR_DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/detector"
BATCH_SIZE = 32
CKPT = "D:/ckpts/capstone/torch/classifier.pth"


def test_rnn_data():
    train_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "train")

    x, y = next(train_loader)

    for i in range(8):
        img = x[14][i]
        img = np.transpose(img, axes=(1, 2, 0))
        print(img.max(), img.min())
        plt.imshow(img)

        print(y[14])

        plt.show()


def test_classifier():
    test_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "test")

    from ai_torch_ver.Classifier import Classifier
    clf = Classifier(4, 0.5).cuda()
    clf.load(CKPT)

    with torch.no_grad():
        test_score = 0.0
        cnt = 0
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            y_batch = torch.max(y_batch, dim=1)[1]
            
            score = clf.score(x_batch, y_batch)
            test_score += score
            cnt += 1

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


test_detector_image()
# test_rnn_data()
# test_classifier()
