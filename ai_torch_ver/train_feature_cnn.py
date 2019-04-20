
import torch
from torch import optim, nn
import numpy as np
import sys

sys.path.append("../")

from ai_torch_ver.features.FeatureCNNv2 import FeatureCNN
from ai_torch_ver.prepare_data import image_loader

VM = False
if VM:
    CKPT = "ckpts/feature_cnn.pth"
    TRASH_DATA_PATH = "/home/jylee/datasets/capstonedata/total/"
else:
    CKPT = "ckpts/feature_cnn.pth"
    # CKPT = "D:/ckpts/capstone/torch/feature_cnn.pth"
    TRASH_DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/total"
ETA = 3e-4
BATCH_SIZE = 128
EPOCHS = 50
DROP_RATE = 0.5
NUM_CLASSES = 4


def score(logps, labels):
    ps = torch.exp(logps)
    cls_ps, top_k = ps.topk(1, dim=1)
    equal = top_k == labels.view(*top_k.shape)
    acc = torch.mean(equal.type(torch.FloatTensor))
    return acc


def train_feature_cnn():
    
    device = torch.device("cuda:0")
    cnn = FeatureCNN(NUM_CLASSES, DROP_RATE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(cnn, device_ids=[0, 1]).to(device)
    else:
        model = cnn.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA)

    val_losses = []

    # train_loader, valid_loader, test_loader = image_loader(TRASH_DATA_PATH, BATCH_SIZE)
    train_loader = image_loader(TRASH_DATA_PATH, BATCH_SIZE, True)

    for e in range(EPOCHS):
        
        train_loss = 0.0
        train_acc = 0.0
        cnt = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logps = model(x_batch)
            loss = criterion(logps, y_batch)

            train_loss += loss.item()
            train_acc += score(logps, y_batch).item()
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epochs {e+1}/{EPOCHS}")
        print(f"Train loss: {train_loss/cnt:.6f}")
        print(f"Train acc: {train_acc/cnt:.6f}")

        # with torch.no_grad():
        #     model.eval()

        #     val_loss = 0.0
        #     val_acc = 0.0

        #     for x_batch, y_batch in valid_loader:
        #         x_batch = x_batch.to(device)
        #         y_batch = y_batch.to(device)

        #         logps = model(x_batch)
        #         loss = criterion(logps, y_batch)

        #         val_loss += loss.item()
        #         val_acc += score(logps, y_batch)

        #     train_loss /= len(train_loader)
        #     train_acc /= len(train_loader)
        #     val_loss /= len(valid_loader)
        #     val_acc /= len(valid_loader)

        #     print(f"Epochs {e+1}/{EPOCHS}")
        #     print(f"Train loss: {train_loss:.6f}")
        #     print(f"Train acc: {train_acc:.6f}")
        #     print(f"Valid loss: {val_loss:.6f}")
        #     print(f"Valid acc: {val_acc:.6f}")

        #     if val_losses and min(val_losses) > val_loss:
        #         if type(model) is nn.DataParallel:
        #             model.module.save(CKPT)
        #         else:
        #             model.save(CKPT)
        #         # state_dict = model.state_dict()
        #         # torch.save(state_dict, CKPT)

        #     val_losses.append(val_loss)
        #     model.train()

    if type(model) is nn.DataParallel:
        model.module.save(CKPT)
    else:
        model.save(CKPT)

if __name__ == "__main__":
    train_feature_cnn()