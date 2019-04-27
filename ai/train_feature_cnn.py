
import torch
from torch import optim, nn
import numpy as np
import sys

sys.path.append("../")

from ai.features.FeatureCNN import FeatureCNN
from ai.prepare_data import image_loader_trash

CKPT = "ckpts/feature_cnn_train2.pth"
# CKPT = "ckpts/feature_cnn.pth"
ETA = 3e-4
BATCH_SIZE = 128
EPOCHS = 80
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
    # cnn.load(CKPT)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(cnn, device_ids=[0, 1]).to(device)
    else:
        model = cnn.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=ETA, weight_decay=1e-3)

    min_val_loss = np.inf

    # train_loader, valid_loader, test_loader = image_loader(TRASH_DATA_PATH, BATCH_SIZE)

    for e in range(EPOCHS):
        # 테스트시에 사진이랑 트레이닝 데이터 사진 RGB별로 평균픽셀값 비교해보기
        # min max normal 말고 standardize가 더 나을 수도.
        
        train_loss = 0.0
        train_acc = 0.0
        cnt = 0
        train_loader = image_loader_trash(BATCH_SIZE, train=True)
        valid_loader = image_loader_trash(BATCH_SIZE, train=False)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logps = model(x_batch)
            loss = criterion(logps, y_batch)

            train_loss += loss.item()
            with torch.no_grad():
                model.eval()
                train_acc += score(logps, y_batch).item()
                model.train()
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= cnt
        train_acc /= cnt

        with torch.no_grad():
            model.eval()

            val_loss = 0.0
            val_acc = 0.0
            cnt = 0

            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logps = model(x_batch)
                loss = criterion(logps, y_batch)

                val_loss += loss.item()
                val_acc += score(logps, y_batch)

                cnt += 1

            val_loss /= cnt
            val_acc /= cnt

            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.6f}")
            print(f"Train acc: {train_acc:.6f}")
            print(f"Valid loss: {val_loss:.6f}")
            print(f"Valid acc: {val_acc:.6f}")

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                if type(model) is nn.DataParallel:
                    model.module.save(CKPT)
                else:
                    model.save(CKPT)
                # state_dict = model.state_dict()
                # torch.save(state_dict, CKPT)

            model.train()

    # if type(model) is nn.DataParallel:
    #     model.module.save(CKPT)
    # else:
    #     model.save(CKPT)


if __name__ == "__main__":
    train_feature_cnn()
