from test_main import train_FMD_cnn, train_trash_cnn
import threading as th

if __name__ == "__main__":
	t1 = th.Thread(target=train_FMD_cnn, args=(0,))
	t2 = th.Thread(target=train_trash_cnn, args=(1,))

	t1.start()
	t2.start()

	t1.join()
	t2.join()
