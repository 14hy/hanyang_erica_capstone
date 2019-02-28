import cv2
import socket
import numpy as np

class Client:

	def __init__(self):
		self.serv_conn = None
		self.capture = False

	def __del__(self):
		if self.serv_conn is not None:
			self.serv_conn.close()

	def connect(self):
		host = "localhost"
		port = 10333
		self.serv_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.serv_conn.connect((host, port))

	def start_capture(self):
		pass

	def test(self):
		cap = cv2.VideoCapture("D:\\Users\\jylee\\Downloads\\KakaoTalk\\1.mp4")
		while True:
			ret, frame = cap.read()
			if frame is None:
				break

			encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
			result, imgencode = cv2.imencode('.jpg', frame, encode_param)

			data = np.array(imgencode)
			strdata = data.tostring()

			size = len(strdata)

			self.serv_conn.send(str(size).ljust(16).encode("utf-8"))
			self.serv_conn.sendall(strdata)


clnt = Client()
clnt.connect()
clnt.test()