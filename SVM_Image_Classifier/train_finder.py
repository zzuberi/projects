import sys
from os.path import join

import cv2
import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn import svm
from sklearn.externals import joblib


def main():
    data = get_data(data_path)
    pos_samps = []
    neg_samps = []
    b_h = 36
    b_w = b_h

    for i, img_path in enumerate(data.index):
        img = cv2.imread(join(data_path, img_path))
        h = img.shape[0]
        w = img.shape[1]
        loc = (int(data.at[img_path, 'x'] * w), int(data.at[img_path, 'y'] * h))

        pos_samps.extend(generate_pos_samps(img, loc, b_h, b_w))
        if len(pos_samps) != len(neg_samps):
            neg_samps.extend(generate_neg_samps(img, loc, b_h, b_w, h, w))

    y = [1] * len(pos_samps)
    y.extend([0] * len(neg_samps))

    X = pos_samps
    X.extend(neg_samps)

    hog = get_hog_descriptor(b_h, b_w)

    X = [hog.compute(x).flatten() for x in X]

    clf = train_svm(X, y)

    joblib.dump(clf, 'classifier.pkl')


def get_data(data_path):
    label_file = join(data_path, 'labels.txt')
    return pd.read_csv(label_file, sep=' ', header=None, names=['x', 'y'], index_col=0)


def get_hog_descriptor(b_h, b_w):
    win_size = (2 * b_h, 2 * b_w)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_threshold = .2
    gamma_correction = True
    nlevels = 64

    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma,
                             histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)


def train_svm(X, y):
    clf = svm.SVC(C=2 ** 5, gamma=2 ** -9)
    clf.fit(X, y)
    return clf


def generate_pos_samps(img, loc, b_h, b_w):
    positive_samples = []
    crop = img[loc[1] - b_h:loc[1] + b_h, loc[0] - b_w:loc[0] + b_w, :]
    if crop.shape == (72, 72, 3):
        positive_samples.append(crop)
        positive_samples.append(cv2.flip(crop, 0))
        positive_samples.append(cv2.flip(crop, 1))
        positive_samples.append(cv2.flip(crop, -1))
    return positive_samples


def generate_neg_samps(img, loc, b_h, b_w, h, w):
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


def eucl_distance(X, Y):
    return np.sqrt(np.inner(X - Y, X - Y))


if __name__ == "__main__":
    global data_path
    data_path = sys.argv[1]
    main()
