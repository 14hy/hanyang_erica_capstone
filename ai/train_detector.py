import torch
from torch import optim, nn
import numpy as np

from TrashDetector import TrashDetector
from prepare_data import image_loader_detector

CKPT = "ckpts/detector3.pth"
ETA = 3e-4
BATCH_SIZE = 128
EPOCHS = 25
DROP_RATE = 0.5


def score(logps, labels):
    ps = torch.exp(logps)
    cls_ps, top_k = ps.topk(1, dim=1)
    equal = top_k == labels.view(*top_k.shape)
    acc = torch.mean(equal.type(torch.FloatTensor))
    return acc


def train_detector_all():
    device = torch.device("cuda")
    detector = TrashDetector(DROP_RATE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(detector, device_ids=[0, 1]).to(device)
    else:
        model = detector.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA, weight_decay=1e-3)

    min_val_loss = np.inf

    for e in range(EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        cnt = 0

        train_loader = image_loader_detector(BATCH_SIZE, train=True)
        valid_loader = image_loader_detector(BATCH_SIZE, train=False)

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logps = model(images)
            loss = criterion(logps, labels)

            train_loss += loss.item()

            ps = torch.exp(logps)
            cls_ps, topk = ps.topk(1, dim=1)
            equal = topk == labels.view(*topk.size())
            train_acc += torch.mean(equal.float())

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

            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                logps = model(images)
                loss = criterion(logps, labels)

                val_loss += loss.item()
                ps = torch.exp(logps)
                cls_ps, topk = ps.topk(1, dim=1)
                equal = topk == labels.view(*topk.size())
                val_acc += torch.mean(equal.float())
                
                cnt += 1

            val_loss /= cnt
            val_acc /= cnt

        print(f"Epochs: {e+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Train acc: {train_acc:.8f}")
        print(f"Valid loss: {val_loss:.8f}")
        print(f"Valid acc: {val_acc:.8f}")

        if min_val_loss > val_loss:
            min_val_loss = val_loss
            if type(model) is nn.DataParallel:
                model.module.save(CKPT)
            else:
                model.save(CKPT)


if __name__ == "__main__":
    # train_detector()
    train_detector_all()
