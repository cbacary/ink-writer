import cv2
import numpy as np
import struct


def read_data(image_file, label_file):
    with open(label_file, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

    with open(image_file, 'rb') as f:
        magic, num, nrows, ncols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        images = images.reshape((num, nrows, ncols))
        # For some reason images are mirrored along x-axis and rotated 90
        for count, image in enumerate(images):
            img = cv2.flip(image, 1)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = 255 - img
            images[count] = img


    return images, labels

def create_map(LBL_FILE):

    MISSING = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']

    map = {}

    with open(LBL_FILE, 'r') as f:
        lines = f.readlines()

    for line in lines:
        l = chr(int(line.split()[1]))
        n = int(line.split()[0])
        map[l] = n

    for i in MISSING:
        map[i] = map[i.upper()]

    return map

def get_word_rank(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    return [line.split()[0] for line in lines]

def show_img(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
