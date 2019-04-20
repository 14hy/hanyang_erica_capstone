
import sys
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from ai_torch_ver.TrashDetector_v2 import TrashDetector
from ai_torch_ver.Classifier import Classifier

THRESHOLD = 0.6


class AI():

    def __init__(self):
        self.classifier = None
        self.trash_detector = None

    def build(self):
        self.classifier = Classifier(num_classes=4, drop_rate=0.5).cuda()
        self.classifier.load("../ai_torch_ver/ckpts/classifier.pth")
        for param in self.classifier.parameters():
            param.requires_grad_(False)

        self.classifier.eval()

        # self.trash_detector = TrashDetector(drop_rate=0.5).cuda()
        self.trash_detector = TrashDetector().cuda()
        # self.trash_detector.load("../ai_torch_ver/ckpts/detector.pth")
        self.trash_detector.load("../ai_torch_ver/ckpts/detector_v2.pth")
        for param in self.trash_detector.parameters():
            param.requires_grad_(False)

        self.trash_detector.eval()

    def predict(self, images):
        # images = np.transpose(images, (3, 1, 2))

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        n = len(images)
        torch_images = []
        for i in range(n):
            img = images[i].astype(np.float32) / 255
            img_tensor = torch.FloatTensor(img).permute(2, 0, 1)
            print(img_tensor.max(), img_tensor.min())
            torch_images.append(img_tensor)

            # import matplotlib.pyplot as plt
            # plt.imshow(img_tensor.permute(1, 2, 0).data)
            # plt.show()

        torch_images = torch.stack(torch_images, dim=0).cuda()
        print(torch_images.size())
        
        with torch.no_grad():
            preds = self.trash_detector.predict(torch_images)
            print(preds)
            if preds == 1:  # self.is_trash(preds):
                print("This is a trash!")
                torch_images = torch_images.unsqueeze(dim=0)
                pred = self.classifier.predict(torch_images)
                return int(pred)
            else:
                print("This is not a trash.")
                return -1

    # def is_trash(self, preds):
    #     if torch.mean(preds.float()) >= THRESHOLD:
    #         return True
    #     else:
    #         return False
