import torch
from torch import optim, nn
import numpy as np

from TrashDetector_v2 import TrashDetector
from prepare_data import get_detector_data

DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/detector"
CKPT = "ckpts/detector_v2.pth"
ETA = 1e-4
BATCH_SIZE = 32
EPOCHS = 350
DEVICE_IDS = [0, 1]
# DROP_RATE = 0.4


def loss_fn(pos, neg):
    triplet_loss = neg + 1 / (pos + 1e-6)
    return triplet_loss


def train_detector():
    
    device = torch.device("cuda:0")
    detector = TrashDetector()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(detector, device_ids=DEVICE_IDS).to(device)
    else:
        model = detector.to(device)

    optimizer = optim.Adam(model.parameters(), lr=ETA)

    for e in range(EPOCHS):
        train_loss = 0.0

        loader = iter(get_detector_data(DATA_PATH, BATCH_SIZE))
        cnt = 0

        for data in loader:
            data = data.to(device)
            pos, neg = detector(data)
            loss = loss_fn(pos, neg)

            train_loss += loss.item()
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= cnt

        print(f"Epochs {e+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.6f}")

    if type(model) is nn.DataParallel:
        model.module.save(CKPT)
    else:
        model.save(CKPT)


if __name__ == "__main__":
    train_detector()