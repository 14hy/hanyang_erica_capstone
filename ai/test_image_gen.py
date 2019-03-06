
import matplotlib.pyplot as plt

from datautils import trash_data_generator, FMD_data_generate

VM = False

def test_trash_data():
	loader = iter(trash_data_generator(VM, 128, "train"))
	images, labels, _ = next(loader)
	images, labels, _ = next(loader)
	images, labels, _ = next(loader)
	images, labels, _ = next(loader)

	for i in range(5):
		plt.imshow(images[i])
		print(labels[0])
		plt.show()

	loader = iter(trash_data_generator(VM, 128, "valid"))
	images, labels, _ = next(loader)
	images, labels, _ = next(loader)
	images, labels, _ = next(loader)

	for i in range(5):
		plt.imshow(images[i])
		print(labels[0])
		plt.show()


def test_FMD_data():
	loader = iter(FMD_data_generate(VM, 128, "train"))
	images, labels = next(loader)
	images, labels = next(loader)
	images, labels = next(loader)
	images, labels = next(loader)

	for i in range(5):
		plt.imshow(images[i])
		print(labels[0])
		plt.show()

	loader = iter(FMD_data_generate(VM, 128, "valid"))
	images, labels = next(loader)
	images, labels = next(loader)
	images, labels = next(loader)

	for i in range(5):
		plt.imshow(images[i])
		print(labels[0])
		plt.show()


# test_trash_data()
test_FMD_data()