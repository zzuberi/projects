from os.path import join

import cv2
import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn import svm


class Finder:
    """
    This class creates a SVM single object identifier from a given set of training images and object locations.

    Training data includes a set of images with the desired object within them.
    Object locations should be given with unit normalized image coordinates (origin considered top left corner of image)
    .
    Object should have approximately similar size in all images.
    """

    def __init__(self, data_path):
        """
        Constructor for Finder class.
        :param data_path: string to the csv file with object locations within image. File format is as follows:
        image_path, x-coordinate, y-coordinate
        """
        self.hog = None
        self.data_path = data_path
        self.clf = None
        self.b_h = None
        self.b_w = None

    def set_hog_descriptor(self, b_h=36, b_w=36, block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9,
                           deriv_aperture=1, win_sigma=-1,
                           histogram_norm_type=0, l2_hys_threshold=.2, gamma_correction=True, nlevels=64):
        """

        :param b_h: desired window height of image detection block
        :param b_w: desired window width of image detection block
        :param block_size: see cv2.HOGDescriptor()
        :param block_stride: see cv2.HOGDescriptor()
        :param cell_size: see cv2.HOGDescriptor()
        :param nbins: see cv2.HOGDescriptor()
        :param deriv_aperture: see cv2.HOGDescriptor()
        :param win_sigma: see cv2.HOGDescriptor()
        :param histogram_norm_type: see cv2.HOGDescriptor()
        :param l2_hys_threshold: see cv2.HOGDescriptor()
        :param gamma_correction: see cv2.HOGDescriptor()
        :param nlevels: see cv2.HOGDescriptor()
        """
        self.b_h = b_h
        self.b_w = b_w
        win_size = (2 * self.b_h, 2 * self.b_w)
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma,
                                     histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)

    def train(self):
        """
        Train SVM with given training data.

        HOG Descriptor must be set prior to training.
        """
        assert self.hog is not None

        data = self.__get_data__()
        pos_samps = []
        neg_samps = []

        for i, img_path in enumerate(data.index):
            img = cv2.imread(join(self.data_path, img_path))
            if img is None:
                continue
            h = img.shape[0]
            w = img.shape[1]
            loc = (int(data.at[img_path, 'x'] * w), int(data.at[img_path, 'y'] * h))

            pos_samps.extend(self.__generate_pos_samps__(img, loc, self.b_h, self.b_w))
            if len(pos_samps) != len(neg_samps):
                neg_samps.extend(self.__generate_neg_samps__(img, loc, self.b_h, self.b_w, h, w))

        y = [1] * len(pos_samps)
        y.extend([0] * len(neg_samps))

        x = pos_samps
        x.extend(neg_samps)

        x = [self.hog.compute(item).flatten() for item in x]

        self.clf = self.__train_svm__(x, y)

    def __get_data__(self):
        label_file = join(self.data_path, 'labels.txt')
        return pd.read_csv(label_file, sep=' ', header=None, names=['x', 'y'], index_col=0)

    def __train_svm__(self, x, y):
        clf = svm.SVC(C=2 ** 5, gamma=2 ** -9)
        clf.fit(x, y)
        return clf

    def __generate_pos_samps__(self, img, loc, b_h, b_w):
        positive_samples = []
        crop = img[loc[1] - b_h:loc[1] + b_h, loc[0] - b_w:loc[0] + b_w, :]
        if crop.shape == (72, 72, 3):
            positive_samples.append(crop)
            positive_samples.append(cv2.flip(crop, 0))
            positive_samples.append(cv2.flip(crop, 1))
            positive_samples.append(cv2.flip(crop, -1))
        return positive_samples

    def __generate_neg_samps__(self, img, loc, b_h, b_w, h, w):
        neg_samps = []
        for i in range(4):
            n_h = randint(0, h)
            n_w = randint(0, w)

            while not (0 <= n_h < loc[1] - 3 * b_h) and not (loc[1] + b_h <= n_h < h - 2 * b_h):
                n_h = randint(0, h)

            while not (0 <= n_w < loc[0] - 3 * b_w) and not (loc[0] + b_w <= n_w < w - 2 * b_w):
                n_w = randint(0, w)

            neg_samps.append(img[n_h:n_h + 2 * b_h, n_w:n_w + 2 * b_w, :])

        return neg_samps

    def evaluate(self, img_path, stride=4):
        """
        Evaulate test image.
        :param img_path: path to image for which objct needs to be identified
        :param stride: desired stride for detection window
        :return: returns numpy array of predicted [x, y]
        """
        assert self.clf is not None

        img = cv2.imread(img_path)
        h = img.shape[0]
        w = img.shape[1]

        blank = np.zeros((h, w))
        for i in range(0, h - 2 * self.b_h, stride):
            for j in range(0, w - 2 * self.b_w, stride):
                prediction = self.clf.predict(
                    self.hog.compute(img[i:i + 2 * self.b_h, j:j + 2 * self.b_w, :]).flatten().reshape(1, -1))
                if prediction[0] == 1:
                    blank[i:i + 2 * self.b_h, j:j + 2 * self.b_w] = blank[i:i + 2 * self.b_h, j:j + 2 * self.b_w] + 1
        temp = np.zeros(blank.shape)
        temp[np.where(blank == max(blank.flatten()))] = 1.0
        temp = np.uint8(temp * 255)
        contours, heirarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        conts = []
        for cont in contours:
            conts.extend(cont)
        rect = cv2.boundingRect(np.array(conts))
        detection = np.array(((rect[0] + rect[2] / 2) / float(w), (rect[1] + rect[3] / 2) / float(h)))

        return np.array((round(detection[0], 4), round(detection[1], 4)))
