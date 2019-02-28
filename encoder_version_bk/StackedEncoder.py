import numpy as np
import tensorflow as tf
import sys

sys.path.append(".")

from Encoder import Encoder

class StackedEncoder():

    def __init__(self):
        self.encoders = []

    def build(self, shapes, device, eta=[1e-2, 3e-2, 5e-2]):
        assert (len(shapes) == len(eta))

        self.eta = eta

        num = len(shapes)

        for i in range(num):
            h, w, c = shapes[i]
            enc = Encoder("/home/jylee/capstone/encoder_version/ckpts/encoder{}.ckpt".format(i + 1), eta=eta[i])
            enc.build((h, w, c), device)
            self.encoders.append(enc)

            print(enc.get_encoded_shape())

    def load_weights(self):
        for enc in self.encoders:
            enc.load()

    def fit(self, X_data, index, overwrite=True, epochs=[100, 120, 140], batch_size=128):
        for i in range(index[0]):
            enc = self.encoders[i]
            enc.load()
        for i in index:
            enc = self.encoders[i]

            X_input = X_data
            for j in range(i):
                X_input = self.encoders[j].encode(X_input)

            enc.fit(X_input, overwrite=overwrite, epochs=epochs[i], batch_size=batch_size)

    def encode(self, X_data):
        encoded = X_data
        for enc in self.encoders:
            encoded = enc.encode(encoded)

        return encoded