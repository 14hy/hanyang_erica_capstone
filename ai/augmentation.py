import Augmentor
import pathlib


image_path = "D:/Users/jylee/Dropbox/Files/Datasets/trashdata2"

EXTRA = 3200
CAN = 1600
GLASS = 2000
PLASTIC = 2400

if __name__ == "__main__":
    for d in pathlib.Path(image_path).glob("*"):
        p = Augmentor.Pipeline(d)
        p.crop_random(probability=0.5, percentage_area=0.9)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.rotate90(probability=0.5)
        p.rotate270(probability=0.5)

        if d.name == "extra":
            p.sample(EXTRA)
        elif d.name == "can":
            p.sample(CAN)
        elif d.name == "glass":
            p.sample(GLASS)
        elif d.name == "plastic":
            p.sample(PLASTIC)

