import torch

from torch import nn

try:
    from features.FeatureCNNv2 import FeatureCNN
except:
    from ai_torch_ver.features.FeatureCNNv2 import FeatureCNN


class SiameseNet(nn.Module):

    def __init__(self, num_classes, drop_rate, beta1=0.5, beta2=0.5):
        super(SiameseNet, self).__init__()

        self.cnn = FeatureCNN(num_classes, drop_rate)
        # try:
        #     self.cnn.load("ckpts/feature_cnn.pth")
        # except:
        #     self.cnn.load("../ai_torch_ver/ckpts/feature_cnn.pth")
        self.criterion = nn.MSELoss()
        self.criterion_clf = nn.NLLLoss()

    def forward(self, x_src, x_pos, x_neg):
        f_src = self.cnn.get_features(x_src)
        f_pos = self.cnn.get_features(x_pos)
        f_neg = self.cnn.get_features(x_neg)

        logps_src = self.cnn(x_src)
        logps_pos = self.cnn(x_pos)
        logps_neg = self.cnn(x_neg)

        return f_src, f_pos, f_neg, logps_src, logps_pos, logps_neg

    def loss_fn(self, x_src, x_pos, x_neg,
                logps_src, logps_pos, logps_neg,
                src_labels, pos_labels, neg_labels):

        pos_mse = self.criterion(x_src, x_pos)
        neg_mse = self.criterion(x_src, x_neg)

        triplet_loss = pos_mse - neg_mse + 0.35

        src_loss = self.criterion_clf(logps_src, src_labels)
        pos_loss = self.criterion_clf(logps_pos, pos_labels)
        neg_loss = self.criterion_clf(logps_neg, neg_labels)

        loss = 0.5*triplet_loss + torch.sum(src_loss + pos_loss + neg_loss)

        return loss

    def save(self, ckpt):
        torch.save(self.state_dict(), ckpt)
        print("SiameseNet was saved.")

    def load(self, ckpt):
        self.load_state_dict(torch.load(ckpt))
        print("SiameseNet was loaded.")





