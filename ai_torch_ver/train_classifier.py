from torchvision import datasets, transforms
import torch
from torch import nn, optim
import sys

sys.path.append(".")

from prepare_data import image_loader, rnn_data
from Classifier import Classifier
# from features.FeatureCNNv2 import FeatureCNN

VM = False
if VM:
    CKPT = "ckpts/classifier.pth"
    TRASH_DATA_PATH = "/home/jylee/datasets/capstonedata/total/"
else:
    CKPT = "D:/ckpts/capstone/torch/classifier.pth"
    TRASH_DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/total"
ETA = 3e-4
BATCH_SIZE = 32
EPOCHS = 5
DROP_RATE = 0.4
NUM_CLASSES = 4

# FEATURE_CNN_CKPT = "D:/ckpts/capstone/torch/feature_cnn.pth"


# def score(logps, labels):
#     ps = torch.exp(logps)
#     cls_ps, top_k = ps.topk(1, dim=1)
#     equal = top_k == labels.view(*top_k.shape)
#     acc = torch.mean(equal.type(torch.FloatTensor))
#     return acc


# def save(clf):
#     model = {
#         "state_dict": clf.state_dict(),
#         "short_term": clf.hidden[0],
#         "long_term": clf.hidden[1]
#     }
#     torch.save(model, CKPT)
#     print("Model was saved.")


# def load_feature_cnn():
#     pcnn = nn.DataParallel(FeatureCNN(NUM_CLASSES, 0.5), device_ids=[0, 1])
    
#     state_dict = torch.load(FEATURE_CNN_CKPT)
#     pcnn.load_state_dict(state_dict)

#     features = None

#     for m in pcnn.modules():
#         if type(m) is FeatureCNN:
#             features = m
#             break

#     return features


def train_classifier():

    device = torch.device("cuda")
    clf = Classifier(NUM_CLASSES, drop_rate=DROP_RATE)
    # clf.load(CKPT)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(clf, device_ids=[0, 1], output_device=0).to(device)
    else:
        model = clf.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA)

    val_losses = []

    for e in range(EPOCHS):
        train_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "train")
        valid_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "valid")
        test_loader = rnn_data(TRASH_DATA_PATH, BATCH_SIZE, "test")

        train_loss = 0.0
        train_acc = 0.0

        cnt = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = torch.max(y_batch.to(device), dim=1)[1]

            logps = model.forward(x_batch, rearange=False)
            loss = criterion(logps, y_batch)
            train_loss += loss.item()
            # train_acc += score(logps, torch.max(y_batch, dim=1)[1])
            if torch.cuda.device_count() > 1:
                train_acc += model.module.score(logps, y_batch)
            else:
                train_acc += model.score(logps, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1

        train_loss /= cnt
        train_acc /= cnt

        with torch.no_grad():
            model.eval()

            val_loss = 0.0
            val_acc = 0.0

            cnt = 0
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = torch.max(y_batch.to(device), dim=1)[1]

                logps = model.forward(x_batch, rearange=False)
                loss = criterion(logps, y_batch)
                val_loss += loss.item()
                # val_acc += score(logps, torch.max(y_batch, dim=1)[1])

                if torch.cuda.device_count() > 1:
                    val_acc += model.module.score(logps, y_batch)
                else:
                    val_acc += model.score(logps, y_batch)

                cnt += 1

            val_loss /= cnt
            val_acc /= cnt

            val_losses.append(val_loss)
            if min(val_losses) == val_loss:
                # save(clf)
                if torch.cuda.device_count() > 1:
                    model.module.save(CKPT)
                else:
                    model.save(CKPT)

            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.6f}")
            print(f"Train acc: {train_acc:.6f}")
            print(f"Valid loss: {val_loss:.6f}")
            print(f"Valid acc: {val_acc:.6f}")

            model.train()


train_classifier()
