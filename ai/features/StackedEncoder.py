import numpy as np
import sys

sys.path.append(".")

from Encoder import Encoder


class StackedEncoder():

	def __init__(self, ckpt_file, device, batch_size=[128, 128, 128], eta=[1e-3, 1e-3, 1e-3]):
		self.device = device
		self.batch_size = batch_size
		self.ckpt_file = ckpt_file
		self.eta = eta
		self.encoders = []
		self.shapes = []

	def build(self, shape, num_stack=3, load_weights=False):

		h, w, c = shape
		self.shapes.append((h, w, c))

		h = int(np.ceil(h / 4))
		w = int(np.ceil(w / 4))
		c = 32
		self.shapes.append((h, w, c))

		h = int(np.ceil(h / 4))
		w = int(np.ceil(w / 4))
		c = 32
		self.shapes.append((h, w, c))

		for i in range(num_stack):
			enc = Encoder("{}{}.ckpt".format(self.ckpt_file, i+1), self.device,
			              self.batch_size[i], self.eta[i])
			enc.build(self.shapes[i], load_weights)

			self.encoders.append(enc)

	def fit(self, generator_fn, index, epochs=[300, 300, 300]):
		for i in range(0, index[0]):
			self.encoders[i].load_weights()

		for i in index:
			enc = self.encoders[i]

			for e in range(epochs[i]):
				generator = iter(generator_fn(128))
				train_loss = 0.0
				cnt = 0
				for X_batch in generator:
					X_encoded = X_batch
					for j in range(i):
						X_encoded = self.encoders[j].encode(X_encoded)
					enc.fit(X_encoded)
					train_loss += enc.compute_loss(X_encoded)
					cnt += 1

				train_loss /= cnt

				print("Epochs {}/{}".format(e+1, epochs[i]))
				print("Train loss: {:.6f}".format(train_loss))

			enc.save()

	def encode(self, X_data):
		X_encoded = X_data

		for i in range(len(self.encoders)):
			X_encoded = self.encoders[i].encode(X_encoded)

		return X_encoded
