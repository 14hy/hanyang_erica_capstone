import cv2
import socket
import numpy as np

class Server:

	def __init__(self):
		self.serv_sock = None
		self.clnt_conn = None
		self.wait = True

	def __del__(self):
		if self.clnt_conn is not None:
			self.clnt_conn.close()
		if self.serv_sock is not None:
			self.serv_sock.close()

	def open(self, host, port=10333):
		self.serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.serv_sock.bind((host, port))
		self.serv_sock.listen(1)

		self.clnt_conn, self.clnt_addr = self.serv_sock.accept()

	def wait_for_image(self):
		size = 0

		temp = self._recvall(16)
		if temp is None:
			return None
		size = int(temp.decode("utf-8"))
		
		strdata = self._recvall(size)
		if strdata is None:
			return None
		data = np.fromstring(strdata, dtype="uint8")
		img = cv2.imdecode(data, 1)

		cv2.imshow("View", img)
		cv2.waitKey(1)

		return img

	def send_result(self, result):
		data = str(result).ljust(16).encode("utf-8")
		self.clnt_conn.sendall(data)

	def _recvall(self, count):
		buf = b""
		while count:
			newbuf = self.clnt_conn.recv(count)
			if not newbuf:
				return None
			buf += newbuf
			count -= len(newbuf)
		return buf


# serv = Server()
# serv.open()

# while True:
# 	if serv.wait_for_image() is None:
# 		break