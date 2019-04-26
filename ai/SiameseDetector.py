import torch

from torch import nn

try:
    from TrashDetector import TrashDetector
except:
    from ai.TrashDetector import TrashDetector


class SiameseDetector(nn.Module):

    def __init__(self, drop_rate, beta1=0.5, beta2=0.5):
        super(SiameseDetector, self).__init__()

        self.detector = TrashDetector(drop_rate)
        try:
            self.detector.load("ckpts/detector.pth")
        except:
            self.detector.load("../ai/ckpts/detector.pth")
        self.criterion = nn.MSELoss()
        self.criterion_clf = nn.NLLLoss()

    def forward(self, x_src, x_pos, x_neg):
        f_src = self.detector.get_features(x_src)
        f_pos = self.detector.get_features(x_pos)
        f_neg = self.detector.get_features(x_neg)

        logps_src = self.detector(x_src)
        logps_pos = self.detector(x_pos)
        logps_neg = self.detector(x_neg)

        return f_src, f_pos, f_neg, logps_src, logps_pos, logps_neg

    def loss_fn(self, x_src, x_pos, x_neg,
                logps_src, logps_pos, logps_neg,
                src_labels, pos_labels, neg_labels):
        pos_mse = self.criterion(x_src, x_pos)
        neg_mse = self.criterion(x_src, x_neg)

        triplet_loss = pos_mse - neg_mse + 0.35

        src_loss = self.criterion_clf(logps_src, src_labels)
        pos_loss = self.criterion_clf(logps_pos, pos_labels)
        neg_loss = self.criterion_clf(logps_pos, neg_labels)

        loss = triplet_loss + src_loss + pos_loss + neg_loss

        return loss

    def save(self, ckpt):
        torch.save(self.state_dict(), ckpt)
        print("SiameseDetector was saved.")

    def load(self, ckpt):
        self.load_state_dict(torch.load(ckpt))
        print("SiameseDetector was loaded.")





