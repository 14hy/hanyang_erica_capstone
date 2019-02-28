import rospy
import socket

class Client():

	def __init__(self):
		self.serv_conn = None

	def __del__(self):
		if self.serv_conn is not None:
			self.serv_conn.close()

	def connect(self, host, port):
		try:
			self.serv_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.serv_conn.connect((host, port))
		except:
			rospy.logerror("CANNOT connect to server!")

	def send_image(self, img):
		encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
		_, imgencode = cv2.imencode('.jpg', img, encode_param)

		img_data = np.array(imgencode)
		data = img_data.tostring()

		sizeinfo = str(len(data)).encode("utf-8")

		self.serv_conn.sendall(sizeinfo)
		self.serv_conn.sendall(data)

	def recv_result(self):
		data = self.serv_conn.recv(16)
		result = int(data.decode("utf-8"))
		return result