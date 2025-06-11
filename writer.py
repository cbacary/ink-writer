import struct

import cv2
import numpy as np

DEFAULT_LETTER_PADDING = 0.005
DEFAULT_LINE_PADDING = 0.01
DEFAULT_LETTER_SHAPE = (28, 28)
SPACING_VARIANCE_FACTOR = 1.


class Writer:
    def __init__(
        self,
        img_dataset_file,
        lbl_dataset_file,
        map_dataset_file,
    ):
        self.images, self.labels = self.read_data(img_dataset_file, lbl_dataset_file)

        self.map = self.create_map(map_dataset_file)

        # end

    def text_to_img(
        self,
        text,
        letter_shape=DEFAULT_LETTER_SHAPE,
        letter_padding=DEFAULT_LETTER_PADDING,
        line_padding=DEFAULT_LINE_PADDING,
        variance_factor=SPACING_VARIANCE_FACTOR,
        augment_img=None,
    ):
        lines = text.split("\n")
        max_letters = np.max([len(line) for line in lines])
        max_spaces = np.max([line.count(" ") for line in lines])

        w_no_padding = self.len_to_img_w(
            max_letters, shape=letter_shape, letter_padding=letter_padding
        )
        h_no_padding = np.ceil(letter_shape[0] * len(lines))

        padding_w = int(np.round(letter_padding * w_no_padding)) // 2
        padding_h = int(np.round(line_padding * h_no_padding))

        # the image width created will be too large, but it doesn't matter because
        # the cropped image is returned anyway.
        img_w = 4 * (w_no_padding + np.ceil((3 * max_letters) * padding_w))
        img_h = 4 * (h_no_padding + np.ceil(padding_h * len(lines)))

        img = np.ones((int(img_h), int(img_w)), dtype=np.uint8) * 255

        img = self.place_lines_on_img(
            img,
            lines,
            letter_shape=letter_shape,
            padding_w=padding_w,
            padding_h=padding_h,
            variance_factor=variance_factor,
        )

        if augment_img is not None:
            return augment_img(img)

        return img

    def place_lines_on_img(
        self,
        img,
        lines,
        padding_h=0,
        padding_w=0,
        letter_shape=DEFAULT_LETTER_SHAPE,
        variance_factor=SPACING_VARIANCE_FACTOR,
        augment_img=None,
    ):
        letter_h, letter_w = letter_shape

        max_x = 0
        current_y = 0

        for line in lines:
            end_x, _ = self.place_line_on_img(
                img,
                line,
                current_y,
                letter_shape=letter_shape,
                padding_w=padding_w,
                variance_factor=variance_factor,
            )

            max_x = max(end_x, max_x)
            variance = self.get_variance(padding_h, variance_factor)
            current_y += letter_h + padding_h + variance 

        return img[:current_y + letter_h, :max_x]

    def place_line_on_img(
        self,
        img,
        line,
        y,
        letter_shape=DEFAULT_LETTER_SHAPE,
        variance_factor=SPACING_VARIANCE_FACTOR,
        padding_w=0,
    ):
        letter_h, letter_w = letter_shape

        space_size = int(np.round(padding_w * 3))

        x = int(np.abs(self.get_variance(space_size, variance_factor)))
        for count, letter in enumerate(line):
            if letter == " " or letter not in self.map.keys():
                variance = self.get_variance(space_size, variance_factor)
                x += space_size + variance
                continue

            letter_img = self.get_letter(letter)

            if letter_h != letter_img.shape[0] or letter_w != letter_img.shape[1]:
                letter_img = cv2.resize(
                    letter_img,
                    (letter_shape[1], letter_shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            _, threshed = cv2.threshold(
                letter_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            black_pixels = np.where(threshed == 0)
            right_most_black_pix = np.max(black_pixels[1])

            y_v = y + self.get_variance(letter_shape[0], variance_factor / 4.)
            y_v = max(0, y_v)
            img[y_v : y_v + letter_h, x : x + letter_w] = letter_img
            variance = self.get_variance(padding_w, variance_factor)
            x += right_most_black_pix
            x += padding_w + variance

        return (x, img)

    def get_variance(self, spacing, factor):
        max_variance = np.ceil(float(spacing) * factor)
        if max_variance == 0:
            return 0
        return np.random.randint(max_variance // 2, max_variance * 2)

    def read_data(self, image_file, label_file):
        with open(label_file, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))

        with open(image_file, "rb") as f:
            magic, num, nrows, ncols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
            images = images.reshape((num, nrows, ncols))
            # For some reason images are mirrored along x-axis and rotated 90
            for count, image in enumerate(images):
                img = cv2.flip(image, 1)
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img = 255 - img
                images[count] = img

        return images, labels

    def create_map(self, LBL_FILE):
        MISSING = [
            "c",
            "i",
            "j",
            "k",
            "l",
            "m",
            "o",
            "p",
            "s",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]

        map = {}

        with open(LBL_FILE, "r") as f:
            lines = f.readlines()

        for line in lines:
            l = chr(int(line.split()[1]))
            n = int(line.split()[0])
            map[l] = n

        for i in MISSING:
            map[i] = map[i.upper()]

        return map

    def len_to_img_w(
        self, n, shape=DEFAULT_LETTER_SHAPE, letter_padding=DEFAULT_LETTER_PADDING
    ):
        w = round(
            (float(n) * float(shape[1]) * -1.0) / (letter_padding * float(n) - 1.0)
        )

        return w

    def width_to_len(self, w, padding_w_percent, letter_shape_x):
        return round(float(w) / float(padding_w_percent * float(w) + letter_shape_x))

    def get_name(self, initial=False):
        if initial:
            initials_count = np.random.choice([2, 3, 4])
            combined_names = self.first_names + self.last_names
            return "".join(
                np.random.choice(name[0] for name in combined_names)
                for _ in range(initials_count)
            )
        else:
            return (
                np.random.choice(self.first_names)
                + " "
                + np.random.choice(self.last_names)
            )

    def place_names(self, inital=False):
        text = self.get_name(inital)

    def get_letter(self, letter):
        # Get the integer representation of this letter according emnist
        letter_mapped = self.map[letter]

        # Contains the indicies where this letter is located
        letter_images = np.where(self.labels == letter_mapped)[0]

        return self.images[np.random.choice(letter_images)]

    def split_lines(self, text, letters_p_line):
        words = text.split(" ")
        lines = []
        line = ""

        words = text.split(" ")
        for word in words:
            line += word + " "
            if len(line) > letters_p_line:
                lines.append(line)
                line = ""
        if len(line) > 0:
            lines.append(line.rstrip())

        return lines
