import torch
from torch import nn, optim

import numpy as np

from features.SiameseNet import SiameseNet
from prepare_data import siamese_data_loader

CKPT = "ckpts/siamese_net.pth"
CKPT_FCNN = "ckpts/feature_cnn_siamese.pth"
NUM_CLASSES = 4
DROP_RATE = 0.5
ETA = 3e-4
EPOCHS = 50
BATCH_SIZE = 32


def train():
    device = torch.device("cuda")
    net = SiameseNet(NUM_CLASSES, DROP_RATE)
    # net.load(CKPT)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(net).to(device)
    else:
        model = net.to(device)

    optimizer = optim.Adam(model.parameters(), lr=ETA)

    min_val_loss = np.inf

    for e in range(EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        cnt = 0
        train_loader = siamese_data_loader(BATCH_SIZE, train=True)
        valid_loader = siamese_data_loader(BATCH_SIZE, train=False)

        for x_src, y_src, x_pos, y_pos, x_neg, y_neg in train_loader:
            x_src = x_src.to(device)
            y_src = y_src.to(device)
            x_pos = x_pos.to(device)
            y_pos = y_pos.to(device)
            x_neg = x_neg.to(device)
            y_neg = y_neg.to(device)

            enc_src, enc_pos, enc_neg, logps_src, logps_pos, logps_neg = model(x_src, x_pos, x_neg)
            if type(model) is nn.DataParallel:
                loss = model.module.loss_fn(
                    enc_src, enc_pos, enc_neg,
                    logps_src, logps_pos, logps_neg,
                    y_src, y_pos, y_neg,
                )
            else:
                loss = model.loss_fn(
                    enc_src, enc_pos, enc_neg,
                    logps_src, logps_pos, logps_neg,
                    y_src, y_pos, y_neg,
                )

            with torch.no_grad():
                src_ps = torch.exp(logps_src)
                pos_ps = torch.exp(logps_pos)
                neg_ps = torch.exp(logps_neg)

                _, topk_src = src_ps.topk(1, dim=1)
                _, topk_pos = pos_ps.topk(1, dim=1)
                _, topk_neg = neg_ps.topk(1, dim=1)

                score = \
                    torch.mean((topk_src == y_src.view(*topk_src.shape)).float()) + \
                    torch.mean((topk_pos == y_pos.view(*topk_pos.shape)).float()) + \
                    torch.mean((topk_neg == y_neg.view(*topk_neg.shape)).float())

                train_acc += score.item() / 3

            train_loss += loss.item()
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

            for x_src, y_src, x_pos, y_pos, x_neg, y_neg in valid_loader:
                x_src = x_src.to(device)
                y_src = y_src.to(device)
                x_pos = x_pos.to(device)
                y_pos = y_pos.to(device)
                x_neg = x_neg.to(device)
                y_neg = y_neg.to(device)

                enc_src, enc_pos, enc_neg, logps_src, logps_pos, logps_neg = model(x_src, x_pos, x_neg)
                if type(model) is nn.DataParallel:
                    loss = model.module.loss_fn(
                        enc_src, enc_pos, enc_neg,
                        logps_src, logps_pos, logps_neg,
                        y_src, y_pos, y_neg,
                    )
                else:
                    loss = model.loss_fn(
                        enc_src, enc_pos, enc_neg,
                        logps_src, logps_pos, logps_neg,
                        y_src, y_pos, y_neg,
                    )

                with torch.no_grad():
                    src_ps = torch.exp(logps_src)
                    pos_ps = torch.exp(logps_pos)
                    neg_ps = torch.exp(logps_neg)

                    _, topk_src = src_ps.topk(1, dim=1)
                    _, topk_pos = pos_ps.topk(1, dim=1)
                    _, topk_neg = neg_ps.topk(1, dim=1)

                    score = \
                        torch.mean((topk_src == y_src.view(*topk_src.shape)).float()) + \
                        torch.mean((topk_pos == y_pos.view(*topk_pos.shape)).float()) + \
                        torch.mean((topk_neg == y_neg.view(*topk_neg.shape)).float())

                    val_acc += score.item() / 3
                
                val_loss += loss.item()
                cnt += 1

            val_loss /= cnt
            val_acc /= cnt

            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Train acc: {train_acc:.8f}")
            print(f"Valid loss: {val_loss:.8f}")
            print(f"Valid acc: {val_acc:.8f}")

            if min_val_loss > val_loss:
                min_val_loss = val_loss

                if type(model) is nn.DataParallel:
                    model.module.save(CKPT)
                    model.module.cnn.save(CKPT_FCNN)
                else:
                    model.save(CKPT)
                    model.cnn.save(CKPT_FCNN)

            model.train()

    # if type(model) is nn.DataParallel:
    #     model.module.save(CKPT)
    #     model.module.cnn.save(CKPT_FCNN)
    # else:
    #     model.save(CKPT)
    #     model.cnn.save(CKPT_FCNN)


if __name__ == "__main__":
    train()
