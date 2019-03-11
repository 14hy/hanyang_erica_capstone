import torch
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import numpy as np

NUM_CLASSES = 4
NUM_STEP = 8


def image_loader(path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(path, transform)
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset_size = len(indices)
    
    test_indices = indices[: int(dataset_size*0.15)]
    valid_indices = indices[int(dataset_size*0.15) : int(dataset_size*0.3)]
    train_indices = indices[int(dataset_size*0.3) :]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(valid_indices))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(test_indices))

    return train_loader, valid_loader, test_loader


def rnn_data(path, batch_size, settype="train"):
    train_loader, valid_loader, test_loader = image_loader(path, 1024)

    loader = train_loader if settype == "train" \
            else valid_loader if settype == "valid" else test_loader

    for x_batch, y_batch in loader:
        for i in range(30):
            imgs = []
            lbls = []

            cnt = 0
            c = 0

            while cnt < 4:
                x_batch_i = x_batch[y_batch.squeeze() == c]

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


        