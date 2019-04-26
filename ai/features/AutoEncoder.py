import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self, input_channel):
        super(AutoEncoder, self).__init__()

        self.rho = 0.15
        
        self.encoder1 = nn.Conv2d(input_channel, 32, (5, 5), stride=1, padding=2)
        self.encoder2 = nn.Sigmoid()
        self.encoder3 = nn.Conv2d(32, 32, (5, 5), stride=1, padding=2)
        self.encoder4 = nn.Sigmoid()
        self.encoder5 = nn.MaxPool2d((2, 2), stride=2, padding=0, return_indices=True)

        self.decoder1 = nn.MaxUnpool2d((2, 2), stride=2, padding=0)
        self.decoder2 = nn.ConvTranspose2d(32, input_channel, (5, 5), stride=1, padding=2)
        self.decoder3 = nn.Sigmoid()

    def forward(self, x):
        # noise = torch.rand_like(x)
        # ones = torch.ones_like(x)
        # zeros = torch.zeros_like(x)
        # noise = torch.where(noise < 0.3, zeros, ones)
        # x = torch.mul(noise, x)

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        encoded, indices = self.encoder5(x)

        # kl = self.kl_divergense(encoded)

        x = self.decoder1(encoded, indices)
        x = self.decoder2(x)
        decoded = self.decoder3(x)

        return encoded, decoded#, kl

    def loss_fn(self, x, decoded, kl):
        return torch.mean((decoded - x)**2) + 0.5*kl

    def kl_divergense(self, encoded):
        kl = torch.mean(self.rho * torch.log(torch.div(self.rho, encoded)) + \
                        (1 - self.rho) * torch.log(torch.div(1 - self.rho, 1 - encoded)))

        return kl

    def save(self, ckpt):
        torch.save(self.state_dict(), ckpt)
        print("AutoEncoder was saved.")

    def load(self, ckpt):
        self.load_state_dict(torch.load(ckpt))
        print("AutoEncoder was loaded.")
