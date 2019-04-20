import torch
from torch import optim, nn
import numpy as np

from TrashDetector import TrashDetector
from prepare_data import image_loader

VM = False
if VM:
    DATA_PATH = "/home/jylee/datasets/caps"
    CKPT = "ckpts/detector.pth"
else:
    DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/nothing_or_trash"
    CKPT = "ckpts/detector.pth"
ETA = 1e-3
BATCH_SIZE = 128
EPOCHS = 3
DROP_RATE = 0.4


def score(logps, labels):
    ps = torch.exp(logps)
    cls_ps, top_k = ps.topk(1, dim=1)
    equal = top_k == labels.view(*top_k.shape)
    acc = torch.mean(equal.type(torch.FloatTensor))
    return acc


def train_detector():
    
    device = torch.device("cuda:0")
    detector = TrashDetector(DROP_RATE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(detector, device_ids=[0, 1]).to(device)
    else:
        model = detector.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA)

    train_loader, valid_loader, test_loader = image_loader(DATA_PATH, BATCH_SIZE)

    val_losses = []

    for e in range(EPOCHS):
        
        train_loss = 0.0
        train_acc = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logps = model(x_batch)
            loss = criterion(logps, y_batch)

            train_loss += loss.item()
            train_acc += score(logps, y_batch).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()

            val_loss = 0.0
            val_acc = 0.0

            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logps = model(x_batch)
                loss = criterion(logps, y_batch)

                val_loss += loss.item()
                val_acc += score(logps, y_batch)

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            val_loss /= len(valid_loader)
            val_acc /= len(valid_loader)

            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.6f}")
            print(f"Train acc: {train_acc:.6f}")
            print(f"Valid loss: {val_loss:.6f}")
            print(f"Valid acc: {val_acc:.6f}")

            val_losses.append(val_loss)

            if np.min(val_losses) >= val_loss:
                if torch.cuda.device_count() > 1:
                    model.module.save(CKPT)
                else:
                    model.save(CKPT)

            model.train()


if __name__ == "__main__":
    train_detector()