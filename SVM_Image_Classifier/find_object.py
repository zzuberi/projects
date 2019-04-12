import sys

import cv2
import numpy as np
from sklearn.externals import joblib

from train_finder import get_hog_descriptor


def main():
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    b_h = 36
    b_w = b_h

    clf = load_classifier()

    hog = get_hog_descriptor(b_h, b_w)

    blank = np.zeros((h, w))
    stride = 4
    for i in range(0, h - 2 * b_h, stride):
        for j in range(0, w - 2 * b_w, stride):
            prediction = clf.predict(hog.compute(img[i:i + 2 * b_h, j:j + 2 * b_w, :]).flatten().reshape(1, -1))
            if prediction[0] == 1:
                blank[i:i + 2 * b_h, j:j + 2 * b_w] = blank[i:i + 2 * b_h, j:j + 2 * b_w] + 1
    temp = np.zeros(blank.shape)
    temp[np.where(blank == max(blank.flatten()))] = 1.0
    temp = np.uint8(temp * 255)
    img2, contours, heirarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = []
    for cont in contours:
        conts.extend(cont)
    rect = cv2.boundingRect(np.array(conts))
    detection = np.array(((rect[0] + rect[2] / 2) / float(w), (rect[1] + rect[3] / 2) / float(h)))

    print(str(round(detection[0], 4))[:6] + ' ' + str(round(detection[1], 4))[:6])


def load_classifier():
    return joblib.load('classifier.pkl')


if __name__ == "__main__":
    global img_path
    img_path = sys.argv[1]
    main()
