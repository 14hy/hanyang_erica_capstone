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
		c = 16
		self.shapes.append((h, w, c))

		h = int(np.ceil(h / 4))
		w = int(np.ceil(w / 4))
		c = 16
		self.shapes.append((h, w, c))

		for i in range(num_stack):
			enc = Encoder("D:/ckpts/{}{}.ckpt".format(self.ckpt_file, i + 1), self.device,
			              self.batch_size[i], self.eta[i])
			enc.build(self.shapes[i], load_weights)

			self.encoders.append(enc)

	def fit(self, X_data, index, epochs=[300, 300, 300]):
		for i in range(0, index[0]):
			self.encoders[i].load_weights()

		for i in index:
			enc = self.encoders[i]

			X_encoded = X_data
			for j in range(i):
				X_encoded = self.encoders[j].encode(X_encoded)

			enc.fit(X_encoded, epochs[i])

	def encode(self, X_data):
		X_encoded = X_data

		for i in range(len(self.encoders)):
			X_encoded = self.encoders[i].encode(X_encoded)

		return X_encoded
