import matplotlib.pyplot as plt
import numpy as np
from features.FeatureCNN import FeatureCNN
import torch
from prepare_data import image_loader_trash
from sklearn.decomposition import PCA

FEATURE_CNN_CKPT = "ckpts/feature_cnn_train3.pth"
DATA_PATH = "data/trash1/train"
BATCH_SIZE = 500


def main():
    with torch.no_grad():
        device = torch.device("cuda")
        cnn = FeatureCNN(4, 0.0).to(device)
        cnn.load(FEATURE_CNN_CKPT)
        cnn.eval()

        loader = iter(image_loader_trash(BATCH_SIZE, True))
        images, labels = next(loader)

        images = images.to(device)
        labels = labels.to(device)

        f = cnn.get_features(images)
        f = f.cpu().numpy().reshape(BATCH_SIZE, -1)

        pca = PCA(n_components=2)
        pca.fit(f)

        f_2 = pca.transform(f)

        l = labels.cpu().numpy()

        for c, lbl, text in zip(['blue', 'red', 'green', 'purple'], [0, 1, 2, 3], ["can", "extra", "glass", "plastic"]):
            plt.scatter(f_2[l == lbl][:, 0], f_2[l == lbl][:, 1], c=c, alpha=0.3, label=text)
        plt.legend()
        plt.show()


main()
