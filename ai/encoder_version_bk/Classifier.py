import sys

sys.path.append(".")

from StackedEncoder import StackedEncoder
from ClassifierRNN import ClassifierRNN

import numpy as np
import threading as th

class Classifier():

    def __init__(self, num_step, num_classes, num_gpus):
        self.num_step = num_step
        self.num_classes = num_classes
        self.encoders = []
        self.classifier_rnn = None
        self.shapes = [(48, 48, 3), (24, 24, 8), (12, 12, 8)]
        self.encoder_shape = (6, 6, 8)
        self.available_gpu = ["/gpu:{}".format(i) for i in range(num_gpus)]

    def build(self, h, w, c, eta=1e-3, batch_size=128, load_weights=False):
        print("Building Classifier...")

        for g in range(len(self.available_gpu)):
            encoder = StackedEncoder()
            encoder.build(self.shapes, self.available_gpu[g], eta=[1e-3, 1e-3, 1e-3])
            self.encoders.append(encoder)

        self.input_size = self.encoder_shape[0] * self.encoder_shape[1] * self.encoder_shape[2]

        self.classifier_rnn = ClassifierRNN("/home/jylee/capstone/encoder_version/ckpts/classifier_rnn.ckpt", eta=eta,
                                            batch_size=batch_size)
        self.classifier_rnn.build(64, self.num_step, self.input_size, self.num_classes)

        for enc in self.encoders:
            enc.load_weights()

        if load_weights:
            self.classifier_rnn.load()

        print("Classifier was built.")

    def fit(self, X_data, Y_data, overwrite=True, epochs=50):
        """
        Argument
        -------------------
        - X_data: (None, num_step, h, w, c)
        - Y_data: (None, num_classes)
        """

        m, s, h, w, c = np.shape(X_data)
        m, num_classes = np.shape(Y_data)

        assert (np.shape(X_data)[0] == np.shape(Y_data)[0])
        assert (s == self.num_step)

        features = self._get_features(X_data)

        self.classifier_rnn.fit(features, Y_data, overwrite, epochs=epochs)

    def predict(self, X_data):
        """
        Argument
        -------------------
        - X_data: (None, num_step, h, w, c)
        """

        features = self._get_features(X_data)

        result = self.classifier_rnn.predict(features)

        return result

    def score(self, X_data, Y_data):
        features = self._get_features(X_data)

        score = self.classifier_rnn.score(features, Y_data)

        return score

    def _get_features(self, X_data):
        """
        Argument
        -------------------
        - X_data: (None, num_step, h, w, c)
        """

        m, s, h, w, c = np.shape(X_data)

        features = np.zeros((m, self.num_step, self.input_size))

        lst = []

        th1 = th.Thread(target=self._run, args=(lst, 0, X_data))
        th2 = th.Thread(target=self._run, args=(lst, 1, X_data))

        th1.start()
        th2.start()

        th1.join()
        th2.join()

        for i, f in enumerate(lst):
            features[:, i, :] = f

        return features

    def _run(self, lst, idx, img):
        encoder = self.encoders[idx]

        for step in range(idx * (self.num_step // 2), (idx + 1) * (self.num_step // 2)):
            features = encoder.encode(img[:, step, :, :, :])
            lst.append(features.reshape(-1, self.input_size))