#!/usr/bin/env python

import rospy
import socket
import cv2
import numpy as np

class Client():

    def __init__(self):
        self.serv_conn = None
        self.ready = True

    def __del__(self):
        if self.serv_conn is not None:
            self.serv_conn.close()

    def connect(self, host, port):
        try:
            self.serv_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.serv_conn.connect((host, port))
            return True
        except:
            rospy.loginfo("CANNOT connect to server.")
            return False

    def send_image(self, image):
        if self.ready is not True:
            return
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, imgencode = cv2.imencode(".jpg", image, encode_param)

        img_data = np.array(imgencode)
        data = img_data.tostring()

        sizeinfo = str(len(data)).ljust(16).encode("utf-8")

        rospy.loginfo("SEND!")

        self.serv_conn.sendall(sizeinfo)
        self.serv_conn.sendall(data)

    def recv_result(self):
        data = self.serv_conn.recv(16)
        result = int(data.decode("utf-8"))
        return result
