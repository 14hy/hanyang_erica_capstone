import sys
import numpy as np

sys.path.append(".")

class AI():

	def __init__(self):
		self.clf = None

	def setup(self, num_step, num_classes):
		"""
		setup classifier.
		create classifier.

		Arguments:
		----------
		:num_step time step
		:num_classes number of classes into which you want image to be classified
		"""

		from Classifier import Classifier

		self.clf = Classifier(num_step=num_step, num_classes=num_classes, net_type="encoder")
		self.clf.build(num_gpu=2)
		self.clf.load_weights()

	def classify(self, images):
		"""
		Arguments:
		----------
		:images (1, num_step, 128, 128, 3)

		Returns:
		--------
		:preds prediction of image
		"""

		assert(len(images.shape) == 5)

		predictions = self.clf.predict(images)
		pred = np.argmax(predictions, axis=1)[0]

		return pred

