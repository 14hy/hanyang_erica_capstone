import rospy
import rospy
import roslib
import cv2
import numpy as np

from Client import Client
from CameraSub import CameraSub

weights = np.array([10, 8, 6, 4, 3,
					2, 1, 1, 1, 1]) / 20

num_step = 8
host = ""
port = 10333

def is_valuable(lst, image):
	arr = np.array(lst)
	rospy.loginfo("Numpy array: {}".format(arr.shape))
	
	diff = np.square(arr - image.reshape(1, 128, 128, 3))
	diff_val = np.sum(np.multiply(diff, weights.reshape(10, 1, 1, 1)))
	
	rospy.loginfo(diff_val)

	return True

def append_image(lst, image):
	if len(lst) >= 10:
		lst.pop(0)

	lst.append(image)

def main(args):
	rospy.init_node("camera_sub")

	rate = rospy.Rate(10)

	sub = CameraSub()
	clnt = Client()

	clnt.connect(host, port)

	image_arr = []

	is_sending = False
	sending_cnt = 0
	cnt = 0

	while not rospy.is_shutdown():
		image = sub.read_image()
		if image is None:
			rospy.loginfo("Image is None. Shutdown.")
			rospy.shutdown()

		if cnt == 3:
			append_image(image_arr, image)

			if sending_cnt != 0:
				clnt.send_image(image)
				sending_cnt += 1
				
				if sending_cnt == num_step:
					sending_cnt = 0

			elif len(image_arr) >= 10:
				if is_valuable(image_arr, image):
					clnt.send_image(image)
					sending_cnt += 1

		cnt += 1
		append_image(image_arr, image)
		rate.sleep()