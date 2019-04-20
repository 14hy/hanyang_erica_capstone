from torchvision import datasets, transforms
import torch
from torch import nn, optim
import sys
import numpy as np

if __name__ == "__main__":
    sys.path.append("../")

from ai.prepare_data import rnn_data2
from ai.ClassifierVGG16 import ClassifierVGG16

CKPT = "ckpts/classifier_vgg16.pth"
ETA = 1e-4
BATCH_SIZE = 32
EPOCHS = 20


def train_classifier():

    device = torch.device("cuda:0")
    clf = ClassifierVGG16().to(device)
    # clf.load(CKPT)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(clf, device_ids=[0, 1], output_device=0).to(device)
    # else:
    #     model = clf.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(clf.parameters(), lr=ETA)

    min_val_loss = np.inf

    for e in range(EPOCHS):
        train_loader = rnn_data2(BATCH_SIZE, train=True)
        valid_loader = rnn_data2(BATCH_SIZE, train=False)

        train_loss = 0.0
        train_acc = 0.0

        cnt = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()
            # print(y_batch)

            logps = clf.forward(x_batch)
            loss = criterion(logps, y_batch)
            train_loss += loss.item()
            # train_acc += score(logps, torch.max(y_batch, dim=1)[1])
            # if torch.cuda.device_count() > 1:
            #     train_acc += model.module.score(logps, y_batch)
            # else:
            #     train_acc += model.score(logps, y_batch)

            with torch.no_grad():
                train_acc += clf.score(logps, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1

        train_loss /= cnt
        train_acc /= cnt

        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            cnt = 0

            clf.eval()

            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).long()

                logps = clf.forward(x_batch)
                loss = criterion(logps, y_batch)
                val_loss += loss.item()

                val_acc += clf.score(logps, y_batch)

                cnt += 1

            val_loss /= cnt
            val_acc /= cnt

            clf.train()

        print(f"Epochs {e+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Train acc: {train_acc:.6f}")
        print(f"Valid loss: {val_loss:.8f}")
        print(f"Valid acc: {val_acc:.6f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss

            if type(clf) is nn.DataParallel:
                clf.module.save(CKPT)
            else:
                clf.save(CKPT)


train_classifier()
