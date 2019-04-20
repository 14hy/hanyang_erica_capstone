import torch
from torch import optim, nn
import numpy as np

from SiameseDetector import SiameseDetector
from prepare_data import siamese_data_loader

DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/detector"
CKPT = "ckpts/siamese_detector.pth"
ETA = 1e-3
BATCH_SIZE = 16
EPOCHS = 20
DROP_RATE = 0.5


def score(logps, labels):
    ps = torch.exp(logps)
    cls_ps, top_k = ps.topk(1, dim=1)
    equal = top_k == labels.view(*top_k.shape)
    acc = torch.mean(equal.type(torch.FloatTensor))
    return acc


def train_detector_all():
    device = torch.device("cuda")
    detector = SiameseDetector(DROP_RATE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(detector, device_ids=[0, 1]).to(device)
    else:
        model = detector.to(device)

    optimizer = optim.Adam(model.parameters(), lr=ETA)

    for e in range(EPOCHS):
        train_loss = 0.0
        cnt = 0

        loader = siamese_data_loader(DATA_PATH, BATCH_SIZE, True)

        for x_src, y_src, x_pos, y_pos, x_neg, y_neg in loader:
            x_src, x_pos, x_neg = x_src.to(device), x_pos.to(device), x_neg.to(device)
            y_src, y_pos, y_neg = y_src.to(device), y_pos.to(device), y_neg.to(device)

            res = model(x_src, x_pos, x_neg)
            if type(model) is nn.DataParallel:
                loss = model.module.loss_fn(
                    *res,
                    y_src, y_pos, y_neg
                )
            else:
                loss = model.loss_fn(
                    *res,
                    y_src, y_pos, y_neg
                )

            train_loss += loss.item()
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= cnt

        print(f"Epochs: {e+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.8f}")

    if type(model) is nn.DataParallel:
        model.module.save(CKPT)
        model.module.detector.save("ckpts/detector.pth")
    else:
        model.save(CKPT)
        model.detector.save("ckpts/detector.pth")


def train_detector():
    
    device = torch.device("cuda:0")
    detector = SiameseDetector(DROP_RATE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(detector, device_ids=[0, 1]).to(device)
    else:
        model = detector.to(device)

    optimizer = optim.Adam(model.parameters(), lr=ETA)

    min_val_loss = np.inf

    for e in range(EPOCHS):
        
        train_loss = 0.0
        cnt = 0

        train_loader, valid_loader, test_loader = siamese_data_loader(DATA_PATH, BATCH_SIZE)

        for x_src, y_src, x_pos, y_pos, x_neg, y_neg in train_loader:
            x_src, x_pos, x_neg = x_src.to(device), x_pos.to(device), x_neg.to(device)
            y_src, y_pos, y_neg = y_src.to(device), y_pos.to(device), y_neg.to(device)

            res = model(x_src, x_pos, x_neg)
            if type(model) is nn.DataParallel:
                loss = model.module.loss_fn(
                    *res,
                    y_src, y_pos, y_neg
                )
            else:
                loss = model.loss_fn(
                    *res,
                    y_src, y_pos, y_neg
                )

            train_loss += loss.item()
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= cnt

        with torch.no_grad():
            model.eval()

            val_loss = 0.0
            cnt = 0
            for x_src, y_src, x_pos, y_pos, x_neg, y_neg in valid_loader:
                x_src, x_pos, x_neg = x_src.to(device), x_pos.to(device), x_neg.to(device)
                y_src, y_pos, y_neg = y_src.to(device), y_pos.to(device), y_neg.to(device)

                res = model(x_src, x_pos, x_neg)
                if type(model) is nn.DataParallel:
                    loss = model.module.loss_fn(
                        *res,
                        y_src, y_pos, y_neg
                    )
                else:
                    loss = model.loss_fn(
                        *res,
                        y_src, y_pos, y_neg
                    )

                val_loss += loss.item()
                cnt += 1

            val_loss /= cnt

            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Valid loss: {val_loss:.8f}")

            if min_val_loss >= val_loss:
                min_val_loss = val_loss
                if torch.cuda.device_count() > 1:
                    model.module.save(CKPT)
                    model.module.detector.save("ckpts/detector.pth")
                else:
                    model.save(CKPT)
                    model.detector.save("ckpts/detector.pth")

            model.train()


if __name__ == "__main__":
    # train_detector()
    train_detector_all()