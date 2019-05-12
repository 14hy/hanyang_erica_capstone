import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn, optim
import sys

if __name__ == "__main__":
    sys.path.append("../")

from ai.prepare_data import rnn_data, rnn_data2
from ai.Classifier import Classifier
# from features.FeatureCNN import FeatureCNN

CKPT = "ckpts/classifier4.pth"
ETA = 3e-4
BATCH_SIZE = 64
EPOCHS = 50
DROP_RATE = 0.5
NUM_CLASSES = 3

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


def train_classifier_all():

    device = torch.device("cuda")
    clf = Classifier(NUM_CLASSES, drop_rate=DROP_RATE).to(device)
    clf.load(CKPT)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(clf, device_ids=[0, 1], output_device=0).to(device)
    # else:
    #     model = clf.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(clf.parameters(), lr=ETA, weight_decay=1e-3)

    min_val_loss = np.inf

    for e in range(EPOCHS):
        train_loader = rnn_data2(BATCH_SIZE, train=True)
		#valid_loader = rnn_data2(BATCH_SIZE, train=False)

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

        print(f"Epochs {e+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Train acc: {train_acc:.6f}")

    clf.save(CKPT)


def train_classifier():

    device = torch.device("cuda")
    clf = Classifier(NUM_CLASSES, drop_rate=DROP_RATE).to(device)
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
            clf.eval()
            val_loss = 0.0
            val_acc = 0.0
            cnt = 0

            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logps = clf.forward(x_batch)
                loss = criterion(logps, y_batch)
                val_loss += loss.item()

                ps = torch.exp(logps)
                cls_ps, topk = ps.topk(1, dim=1)
                equal = topk == y_batch.view(*topk.shape)
                val_acc += torch.mean(equal.float())

                cnt += 1

            val_loss /= cnt
            val_acc /= cnt

            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Train acc: {train_acc:.6f}")
            print(f"Valid loss: {val_loss:.8f}")
            print(f"Valid acc: {val_acc:.6f}")

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                clf.save(CKPT)

            clf.train()


if __name__ == "__main__":
	train_classifier()
