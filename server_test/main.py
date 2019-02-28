
import sys
import numpy as np
from Server import Server

host = "localhost"
port = 10333
num_step = 8
num_classes = 4
image_width = 128
image_height = 128
image_channel = 3

sys.path.append("../")

from ai.AI import AI

def main(args):
	server = Server()
	server.open(host, port)

	ai = AI()
	ai.setup(num_step, num_classes)

	try:
		while True:
			image_arr = np.zeros((1, num_step, image_height, image_width, image_channel))
			cnt = 0
			
			while cnt < num_step:
				image = server.wait_for_image()
				if image is None:
					continue

				image_arr[0, cnt] = image
				cnt += 1

			prediction = ai.classify(image_arr)
			print(prediction)

			server.send_result(prediction)

	except:
		print("Exception occurs. Server shutdown.")


if __name__ == "__main__":
	main(sys.argv)