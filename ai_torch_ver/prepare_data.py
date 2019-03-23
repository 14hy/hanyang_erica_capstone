import torch
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import cv2

NUM_CLASSES = 4
NUM_STEP = 8


def image_loader(path, batch_size, total=False):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(path, transform)
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset_size = len(indices)
    
    if total is False:
        test_indices = indices[: int(dataset_size*0.70)]
        valid_indices = indices[int(dataset_size*0.70) : int(dataset_size*0.85)]
        train_indices = indices[int(dataset_size*0.85) :]

        train_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(train_indices))
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(valid_indices))
        test_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(test_indices))

        return train_loader, valid_loader, test_loader
    
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
        return loader


def rnn_data(path, batch_size, settype="train", total=False):
    if total is False:
        train_loader, valid_loader, test_loader = image_loader(path, 1024)

        loader = train_loader if settype == "train" \
                else valid_loader if settype == "valid" else test_loader
    else:
        loader = image_loader(path, 1024, total)

    for x_batch, y_batch in loader:
        for i in range(30):
            imgs = []
            lbls = []

            cnt = 0
            c = 0

            while cnt < 4:
                indices = np.arange(y_batch.size(0))
                indices = indices[y_batch.numpy() == c]
                x_batch_i = x_batch[indices]

                r = np.random.choice(np.arange(x_batch_i.shape[0]), size=(batch_size,))

                imgs.append(x_batch_i[r])   
                lbls.append(torch.tensor([c] * (batch_size // NUM_STEP)))

                cnt += 1
                c += 1

                if c == 4:
                    c = 0

            x = torch.stack(imgs, dim=0).view((batch_size * 4) // NUM_STEP, NUM_STEP, 3, 128, 128)
            y = torch.stack(lbls, dim=0).view((batch_size * 4) // NUM_STEP, 1)

            yield x, y


def get_image_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor()
    ])

    dataset = torch.utils.data.ImageFolder(data_dir, transform=transform)

    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_indices = indices[:int(n*0.7)]
    valid_indices = indices[int(n*0.7):int(n*0.85)]
    test_indices = indices[int(n*0.85):]

    train_loader = torch.utils.data.DataLoader(dataset, sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(dataset, sampler=SubsetRandomSampler(valid_indices))
    test_loader = torch.utils.data.DataLoader(dataset, sampler=SubsetRandomSampler(test_indices))

    return train_loader, valid_loader, test_loader


def get_detector_data(data_dir, batch_size):
    nothing_images = []
    trash_images = []
    for d in pathlib.Path(data_dir).glob("*"):
        if d.name == "nothing":
            for f in d.glob("*.jpg"):
                nothing_images.append(f)
        elif d.name == "trash":
            for f in d.glob("*.jpg"):
                trash_images.append(f)

    np.random.shuffle(nothing_images)
    np.random.shuffle(trash_images)

    n = min(len(nothing_images), len(trash_images))
    num_batch = int(np.ceil(n / batch_size))

    for b in range(num_batch):
        start = b * batch_size
        end = min((b+1) * batch_size, n)

        nothing = nothing_images[start:end]
        trash = trash_images[start:end]

        sz = min(batch_size, (end - start) * (end - start - 1))

        source = []
        positive = []
        negative = []

        for i, j, k in zip(np.random.randint(0, end-start, size=(sz,)),
                           np.random.randint(0, end-start, size=(sz,)),
                           np.random.randint(0, end-start, size=(sz,))):
            source_img = cv2.resize(plt.imread(trash[i]), dsize=(128, 128)).astype(np.float32) / 255
            source_img = torch.FloatTensor(source_img.reshape(1, 1, *source_img.shape))
            source_img = source_img.permute(0, 1, 4, 2, 3)

            pos_img = cv2.resize(plt.imread(trash[j]), dsize=(128, 128)).astype(np.float32) / 255
            pos_img = torch.FloatTensor(pos_img.reshape(1, 1, *pos_img.shape))
            pos_img = pos_img.permute(0, 1, 4, 2, 3)

            neg_img = cv2.resize(plt.imread(nothing[k]), dsize=(128, 128)).astype(np.float32) / 255
            neg_img = torch.FloatTensor(neg_img.reshape(1, 1, *neg_img.shape))
            neg_img = neg_img.permute(0, 1, 4, 2, 3)

            source.append(source_img)
            positive.append(pos_img)
            negative.append(neg_img)

        data = torch.cat([
            torch.cat(source, dim=0),
            torch.cat(positive, dim=0),
            torch.cat(negative, dim=0)
        ], dim=1)

        yield data
