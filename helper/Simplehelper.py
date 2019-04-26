import cv2
import threading as th
import os

ok = True

data_path = "D:\\Users\\jylee\\Dropbox\\Files\\Datasets\\trashdata"
# image_path = os.path.join(data_path, "images")
# label_path = os.path.join(data_path, "labels")
labels_dict = ["nothing", "plastic", "can", "glass", "extra"]

def capture(cap):
	while ok:
		_, frame = cap.read()
		cv2.imshow("Helper", frame)
		cv2.waitKey(10)

def main(args):
	cap = cv2.VideoCapture(0)

	cur_index = 1

	with open(os.path.join(data_path, "info.txt"), "r") as f:
		cur_index = int(f.read())

	# label_file = open(os.path.join(label_path, "labels.txt"), "a+")
	
	thread = th.Thread(target=capture, args=(cap,))
	thread.start()

	while True:
		print("==========================")
		print("0. NOTHING")
		print("1. plastic")
		print("2. can")
		print("3. glasses")
		print("4. extra")
		ch = input("=> Label('e' to exit):")
		if ch == "e":
			break
		elif ch.isdigit():
			label = int(ch)
			if not (0 <= label <= 4):
				continue
			_, image = cap.read()

			image = cv2.resize(image, dsize=(128, 128))
			cv2.imwrite(os.path.join(data_path, "{}/%08d.jpg".format(labels_dict[label])) % cur_index, image)
			# label_file.write("{0} {1}/%08d.jpg\n".format(label, labels_dict[label]) % cur_index)

			cur_index += 1

	with open(os.path.join(data_path, "info.txt"), "w") as f:
		f.write(str(cur_index))

	# label_file.close()
	
	global ok
	ok = False

if __name__ == "__main__":
	import sys
	main(sys.argv)