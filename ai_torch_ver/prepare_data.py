import torch
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import numpy as np

def image_loader(path, batch_size):
	transform = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomRotation(90),
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

