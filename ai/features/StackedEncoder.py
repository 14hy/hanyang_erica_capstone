import torch
from torch import nn
try:
    from features.AutoEncoder import AutoEncoder
except:
    from ai_torch_ver.features.AutoEncoder import AutoEncoder

# CKPTS = [
#     "ckpts/encoder1.pth",
#     "ckpts/encoder2.pth",
#     "ckpts/encoder3.pth",
# ]


class StackedEncoder(nn.Module):

    def __init__(self, trainable=[True, True, True]):
        super(StackedEncoder, self).__init__()

        self.encoder1 = AutoEncoder(3)
        self.encoder2 = AutoEncoder(32)
        self.encoder3 = AutoEncoder(32)

        self.encoders = [
            self.encoder1,
            self.encoder2,
            self.encoder3,
        ]

        for i in range(3):
            if not trainable[i]:
                for param in self.encoders[i].parameters():
                    param.requires_grad_(False)
                # self.encoders[i].load(CKPTS[i])

    def forward(self, x, index=2):
        encoded = x

        for i in range(0, index+1):
            encoded, decoded, _ = self.encoders[i](encoded)

        return encoded, decoded

    def train_step(self, x, index, optimizer):
        origin = x
        encoded = None
        decoded = None
        kl = None

        with torch.no_grad():
            for i in range(0, index):
                encoded, decoded, kl = self.encoders[i](origin)
                origin = encoded

        encoded, decoded, kl = self.encoders[index](origin)
        loss = self.encoders[index].loss_fn(origin, decoded, kl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def save(self, ckpt):
        torch.save(self.state_dict(), ckpt)
        print("Stacked encoder was saved.")

    def load(self, ckpt):
        self.load_state_dict(torch.load(ckpt))
        print("Stacked encoder was loaded.")
