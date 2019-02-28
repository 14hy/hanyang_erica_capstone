
import cv2
import numpy as np

class CameraSub():

	def __init__(self):
		self.cap = cv2.VideoCapture(0)

	def read_image(self):
		_, image = self.cap.read()
		image = cv2.resize(image, dsize=(128, 128)).astype(np.float32) / 255
		return image