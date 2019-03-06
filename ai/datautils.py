
import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

def trash_data_generator(VM, batch_size, dataset_type="train"):
	if dataset_type == "train":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/train"
		if VM:
			DATA_DIR = "/home/jylee/datasets/capstonedata/train"
	elif dataset_type == "valid":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/valid"
		if VM:
			DATA_DIR = "/home/jylee/datasets/capstonedata/valid"
	elif dataset_type == "test":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/test"
		if VM:
			DATA_DIR = "/home/jylee/datasets/capstonedata/test"
	else:
		print("Invalid dataset type")
		return

	label_dict = {
		"can": 0,
		"extra": 1,
		"glass": 2,
		"plastic": 3
	}

	data = []

	for d in pathlib.Path(DATA_DIR).glob("*"):
		# print(d)
		if d.name in label_dict.keys():
#print(d.name)
			label = label_dict[d.name]

			for f in d.glob("*.jpg"):
				data.append((f, label))

	# np.random.shuffle(data)
	num_batch = int(np.ceil(len(data) / batch_size))

	for b in range(num_batch):
		start = b * batch_size
		end = min((b+1) * batch_size, len(data))

		X_batch = np.zeros((end - start, 128, 128, 3))
		Y_batch = np.zeros((end - start, len(label_dict.keys())))

		for i in range(start, end):
			img = cv2.resize(plt.imread(data[i][0]), dsize=(128, 128)).astype(np.float32) / 255
			lbl = int(data[i][1])

			X_batch[i - start] = img
			Y_batch[i - start, lbl] = 1

		yield X_batch, Y_batch, num_batch


def FMD_data_generate(VM, batch_size, dataset_type="train"):
	if dataset_type == "train":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image/train"
		if VM:
			DATA_DIR = "/home/jylee/datasets/FMD/image/train"
	elif dataset_type == "valid":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image/valid"
		if VM:
			DATA_DIR = "/home/jylee/datasets/FMD/image/valid"
	elif dataset_type == "test":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image/test"
		if VM:
			DATA_DIR = "/home/jylee/datasets/FMD/image/test"

	label_dict = {
		"fabric": 0,
		"glass": 1,
		"leather": 2,
		"metal": 3,
		"paper": 4,
		"plastic": 5
	}

	data = []

	for d in pathlib.Path(DATA_DIR).glob("*"):
		if d.name in label_dict.keys():
			label = label_dict[d.name]

			for f in d.glob("*.jpg"):
				data.append((f, label))

	num_batch = int(np.ceil(len(data) / batch_size))
	np.random.shuffle(data)

	for b in range(num_batch):
		start = b * batch_size
		end = min((b+1) * batch_size, len(data))

		X_batch = np.zeros((end - start, 128, 128, 3))
		Y_batch = np.zeros((end - start, len(label_dict)))

		for i in range(start, end):
			img = cv2.resize(plt.imread(data[i][0]), dsize=(128, 128)).astype(np.float32) / 255

			if len(img.shape) == 2:
				X_batch[i - start, :, :, 0] = img
				X_batch[i - start, :, :, 1] = img
				X_batch[i - start, :, :, 2] = img
			else:
				X_batch[i - start] = img

			Y_batch[i - start, data[i][1]] = 1

		yield X_batch, Y_batch

def FMD_data_generate_no_label(VM, batch_size):
	if VM:
		DATA_DIR = "/home/jylee/datasets/FMD/image"
	else:
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image"

	data = []

	for f in pathlib.Path(DATA_DIR).glob("*/*/*.jpg"):
		data.append(f)

	num_batch = int(np.ceil(len(data) / batch_size))
	np.random.shuffle(data)

	for b in range(num_batch):
		start = b * batch_size
		end = min((b+1) * batch_size, len(data))

		X_batch = np.zeros((end - start, 128, 128, 3))

		for i in range(start, end):
			img = cv2.resize(plt.imread(data[i]), dsize=(128, 128)).astype(np.float32) / 255

			if len(img.shape) == 2:
				X_batch[i - start, :, :, 0] = img
				X_batch[i - start, :, :, 1] = img
				X_batch[i - start, :, :, 2] = img
			else:
				X_batch[i - start] = img

		yield X_batch


def rnn_trash_data_generator(VM, num_sample, num_step, dataset_type="train"):
	BATCH_SIZE = 512

	assert(num_sample % 4 == 0)
	
	loader = iter(trash_data_generator(BATCH_SIZE, dataset_type))

	for X_batch, Y_batch, num_batch in loader:

		for i in range(4):

			X_batch_category = X_batch[(Y_batch == i).squeeze()]
			if len(X_batch_category) < 32:
				continue

			# print(X_batch_category.shape)

			X_batch_rnn = np.zeros((num_sample, num_step, 128, 128, 3))
			Y_batch_rnn = np.zeros((num_sample, 4))

			r = np.random.randint(0, len(X_batch_category), size=(num_step * (num_sample)))

			X_batch_rnn[:] = X_batch_category[r].reshape(num_sample, num_step, 128, 128, 3)
			Y_batch_rnn[:, i] = 1

			yield X_batch_rnn, Y_batch_rnn, num_batch*4
