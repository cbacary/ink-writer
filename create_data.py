import cv2
import numpy as np
from helper import create_map, get_word_rank, read_data, show_img
from matplotlib import pyplot as plt
from names_dataset import NameDataset, NameWrapper
from writer import Writer


def augment_image(img):
    _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

    count = np.count_nonzero(mask == 255)
    mean = np.random.uniform(0.3, 0.45)
    scale = np.random.uniform(0.10, 0.175)
    normal_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    m = np.random.normal(loc=mean, scale=scale, size=count)
    normal_map[mask == 255] = m
    normal_map = np.clip(normal_map, 0.04, 1.0)

    contrasted = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    contrasted[mask == 255] = np.round(
        contrasted[mask == 255] * normal_map[mask == 255]
    ).astype(np.uint8)
    contrasted[np.where(contrasted <= 11)] = np.random.randint(
        0, 15, size=contrasted[np.where(contrasted <= 11)].shape[0]
    )

    factor = 500.0 / img.shape[0]
    resized = (
        np.copy(contrasted)
        if factor <= 1.0
        else cv2.resize(
            contrasted,
            (int(contrasted.shape[1] * factor), int(contrasted.shape[0] * factor)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    )

    kernel = (
        np.ones((9, 9), np.uint8) if img.shape[0] < 140 else np.ones((3, 3), np.uint8)
    )

    dilated = cv2.dilate(resized, kernel, iterations=2)

    og_size = cv2.resize(
        dilated, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )

    diameter = 5
    s_color = np.random.randint(50, 75)
    s_space = np.random.randint(50, 100)
    filtered = cv2.bilateralFilter(og_size, diameter, s_color, s_space)

    final = cv2.normalize(
        filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    return final


DEFAULT_LETTER_PADDING = 0.01
DEFAULT_LINE_PADDING = 0.01
DEFAULT_LETTER_SHAPE = (28, 28)
MIN_PARAGRAPH_LEN = 20


class DataLoader:
    def __init__(
        self,
        img_dataset_file,
        lbl_dataset_file,
        map_dataset_file,
        wordrank_file,
        N=500,
        writer=None,
    ):
        self.N = N

        if not writer:
            writer = Writer(img_dataset_file, lbl_dataset_file, map_dataset_file)

        self.writer = writer

        self.words = self.get_word_rank(wordrank_file)

        self.nd = NameDataset()
        self.first_names_MF = self.nd.get_top_names(
            n=self.N, country_alpha2="US", use_first_names=True
        )["US"]

        self.first_names = self.first_names_MF["M"] + self.first_names_MF["F"]

        self.last_names = self.nd.get_top_names(
            n=self.N, country_alpha2="US", use_first_names=False
        )["US"]

        self.first_and_last = self.first_names + self.last_names

        self.large_image_sizes = np.random.normal(loc=400.0, scale=40.0, size=self.N)

        self.num_lines = np.random.normal(loc=4.0, scale=1.0, size=self.N)

        self.line_padding_dist = np.random.normal(
            loc=DEFAULT_LINE_PADDING * 2, scale=DEFAULT_LINE_PADDING * 0.6, size=self.N
        )

        self.letter_padding_dist = np.random.normal(
            loc=DEFAULT_LETTER_PADDING, scale=DEFAULT_LETTER_PADDING * 0.1, size=self.N
        )

        self.letter_shape_sizes = np.random.normal(loc=62.0, scale=10.0, size=self.N)

        self.large_image_sizes = np.clip(self.large_image_sizes, 100, None)
        self.line_padding_dist = np.clip(
            self.line_padding_dist, DEFAULT_LINE_PADDING, None
        )
        self.num_lines = np.clip(self.num_lines, 1, None).astype(np.uint8)
        self.letter_padding_dist = np.clip(self.letter_padding_dist, 0.0, None)
        self.letter_shape_sizes = np.clip(self.letter_shape_sizes, 20.0, None)

        # end

    def create_random_paragraph(self, augment=False):
        img_width = np.random.choice(self.large_image_sizes, replace=False)

        lines_n = np.random.choice(self.num_lines, replace=False)

        line_padding = np.random.choice(self.line_padding_dist, replace=False)
        letter_padding = np.random.choice(self.letter_padding_dist, replace=False)
        _letter_shape = round(np.random.choice(self.letter_shape_sizes, replace=False))
        letter_shape = (int(_letter_shape * 1.5), _letter_shape)

        max_letters = self.writer.width_to_len(
            img_width, letter_padding, DEFAULT_LETTER_SHAPE[1]
        )

        soft_max_letters = max_letters
        text = ""
        lines = []
        for i in range(lines_n):
            line = ""
            while len(line) < soft_max_letters:
                word = np.random.choice(self.words)
                for letter in word:
                    line += (
                        letter.lower()
                        if np.random.randint(1, 100) < 50
                        else letter.upper()
                    )
                line += " "
                max_letters = max(max_letters, len(line))
            text += line + "\n"
            lines.append(line)
        text = text[:-1]

        img = self.writer.text_to_img(
            text,
            letter_padding=letter_padding,
            line_padding=line_padding,
            letter_shape=letter_shape,
            augment_img=augment_image if augment else None,
        )

        return img

    def create_random_name(self, initial=False, augment=False):
        letter_padding = np.random.choice(self.letter_padding_dist, replace=False)
        _letter_shape = round(np.random.choice(self.letter_shape_sizes, replace=False))
        letter_shape = (_letter_shape, _letter_shape)

        text = self.get_name(initial=initial)

        img = self.writer.text_to_img(
            text,
            letter_padding=letter_padding,
            letter_shape=letter_shape,
            augment_img=augment_image if augment else None,
        )

        return img

    def create_random_digits(self, augment=False):
        letter_padding = np.random.choice(self.letter_padding_dist, replace=False)
        _letter_shape = round(np.random.choice(self.letter_shape_sizes, replace=False))
        letter_shape = (_letter_shape, _letter_shape)

        text = self.get_digits()

        img = self.writer.text_to_img(
            text,
            letter_padding=letter_padding,
            letter_shape=letter_shape,
            augment_img=augment_image if augment else None,
        )

        return img

    def get_name(self, initial=False):
        result = ""
        if initial:
            initials_count = np.random.choice([2, 3, 4])
            initials = ""
            for _ in range(initials_count):
                result += np.random.choice(self.first_and_last)[0]
        else:
            result = (
                np.random.choice(self.first_names)
                + " "
                + np.random.choice(self.last_names)
            )

        final = ""
        for count, letter in enumerate(result):
            final += (
                letter.lower() if np.random.randint(1, 100) < 50 else letter.upper()
            )

        return result

    def get_digits(
        self,
    ):
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        return "".join(
            [str(i) for i in np.random.choice(digits, size=np.random.randint(4, 16))]
        )

    def get_word_rank(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        return [line.split()[0] for line in lines]


if __name__ == "__main__":

    IMG_FILE = "./dataset/emnist_source_files/emnist-balanced-train-images-idx3-ubyte"
    LBL_FILE = "./dataset/emnist_source_files/emnist-balanced-train-labels-idx1-ubyte"
    MAP_FILE = "./dataset/emnist-balanced-mapping.txt"
    WORDRANK = "./wordrank.txt"

    data_loader = DataLoader(IMG_FILE, LBL_FILE, MAP_FILE, wordrank_file=WORDRANK)

    loader = DataLoader(IMG_FILE, LBL_FILE, MAP_FILE, WORDRANK)

    N_PARAGRAPHS = 125
    N_NAMES = 150
    N_INITIALS = 175
    N_DIGITS = 50

    def write_img(img, path):
        cv2.imwrite(path, img)

    for i in range(N_PARAGRAPHS):
        img = loader.create_random_paragraph(augment=True)
        path = f"./imgs/paragraph_{i}.png"
        write_img(img, path)
    for i in range(N_NAMES):
        img = loader.create_random_name(augment=True, initial=False)
        path = f"./imgs/name_{i}.png"
        write_img(img, path)
    for i in range(N_INITIALS):
        img = loader.create_random_name(augment=True, initial=True)
        path = f"./imgs/initial_{i}.png"
        write_img(img, path)
    for i in range(N_DIGITS):
        img = loader.create_random_digits(augment=True)
        path = f"./imgs/digits_{i}.png"
        write_img(img, path)
