
import sys
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from ai.TrashDetector import TrashDetector
from ai.ClassifierVGG16 import ClassifierVGG16

THRESHOLD = 0.6


class AI():

    def __init__(self):
        self.classifier = None
        self.trash_detector = None
        self.device = torch.device("cuda:0")

    def build(self):
        self.classifier = ClassifierVGG16().to(device)
        self.classifier.load("../ai/ckpts/classifier.pth")
        for param in self.classifier.parameters():
            param.requires_grad_(False)

        self.classifier.eval()

        # self.trash_detector = TrashDetector(drop_rate=0.5).cuda()
        self.trash_detector = TrashDetector(0.0).cuda()
        # self.trash_detector.load("../ai/ckpts/detector.pth")
        self.trash_detector.load("../ai/ckpts/detector.pth")
        for param in self.trash_detector.parameters():
            param.requires_grad_(False)

        self.trash_detector.eval()

    def predict(self, images):
        images = (images.astype(np.float32) - 128) / 256 + 0.5
        print(images.shape, images.min(), images.max())

        torch_images = torch.FloatTensor(images).cuda()
        
        with torch.no_grad():
            # preds = self.trash_detector.predict(torch_images)
            # print(preds)
            # if preds.float().mean() > THRESHOLD:  # self.is_trash(preds):
            #     print("This is a trash!")
            #     torch_images = torch_images.unsqueeze(dim=0)
            #     pred = self.classifier.predict(torch_images)
            #     return int(pred)
            # else:
            #     print("This is not a trash.")
            #     return -1

            torch_images = torch_images.unsqueeze(dim=0)
            pred = self.classifier.predict(torch_images)
            return int(pred)

    # def is_trash(self, preds):
    #     if torch.mean(preds.float()) >= THRESHOLD:
    #         return True
    #     else:
    #         return False
