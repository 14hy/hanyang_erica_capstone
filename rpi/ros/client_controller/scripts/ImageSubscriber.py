#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from std_msgs.msg import UInt8MultiArray

HEIGHT = 128
WIDTH = 128
CHANNEL = 3
NUM_STEP = 8

class ImageSubscriber():

    def __init__(self):
        self.sub = rospy.Subscriber("image_data", UInt8MultiArray, self.image_callback, queue_size=None)
        self.image_data = []
        self.ready = True
        self.cnt = 0

    def get_image(self):
        if len(self.image_data) == 0:
            return None

        img = self.image_data.pop(0)
        return img

    def image_callback(self, data):

        if self.ready is False:
            return

        rospy.loginfo("image received.")
        image = np.zeros((HEIGHT * WIDTH * CHANNEL,))

        for i, c in enumerate(data.data):
            image[i] = ord(c)

        self.image_data.append(image.reshape(HEIGHT, WIDTH, CHANNEL))
        self.cnt += 1

        if self.cnt == 8:
            self.ready = False
