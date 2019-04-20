from features.StackedEncoder import StackedEncoder
from prepare_data import image_loader
import torch
from torch import nn, optim

EPOCHS = [60, 60, 50]
BATCH_SIZE = 128
DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/detector"
CKPT = "ckpts/encoder.pth"


def train_encoder():
    loader = image_loader(DATA_PATH, BATCH_SIZE, total=True)

    device = torch.device("cuda")

    enc = StackedEncoder()
    enc = enc.to(device)

    optimizer = optim.Adam(enc.parameters(), lr=1e-2)

    for i in range(3):
        for e in range(EPOCHS[i]):
            train_loss = 0.0
            cnt = 0

            for images, _ in loader:
                images = images.to(device)

                train_loss += enc.train_step(images, i, optimizer)
                cnt += 1

            train_loss /= cnt

            print(f"Epochs {e+1}/{EPOCHS[i]}")
            print(f"Train loss: {train_loss:.6f}")

    enc.save(CKPT)


if __name__ == "__main__":
    train_encoder()
