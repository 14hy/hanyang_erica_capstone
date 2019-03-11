import numpy as np
import torch
import matplotlib.pyplot as plt
from prepare_data import rnn_data

TRASH_DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/total"
BATCH_SIZE = 32
CKPT = "D:/ckpts/capstone/torch/classifier.pth"


def test_rnn_data():
    train_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "train")

    x, y = next(train_loader)

    for i in np.random.randint(0, x.shape[0], size=(10,)):
        img = x[i][0]
        img = np.transpose(img, axes=(1, 2, 0))
        plt.imshow(img)

        print(y[i][0])

        plt.show()


def test_classifier():
    test_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "test")

    from Classifier import Classifier
    clf = Classifier(4, 0.5).cuda()
    clf.load(CKPT)

    with torch.no_grad():
        test_score = 0.0
        cnt = 0
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            y_batch = torch.max(y_batch, dim=1)[1]
            
            score = clf.score(x_batch, y_batch, False)
            test_score += score
            cnt += 1

        test_score /= cnt

        print(f"Test score: {test_score:.6f}")


# test()
test_classifier()
