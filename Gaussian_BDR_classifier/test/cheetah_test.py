import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from scipy.io import loadmat

from classifier import Classifier

TrainingDCT = loadmat("test_resources/TrainingSamplesDCT_8_new.mat")

TrainsampleDCT_FG = TrainingDCT['TrainsampleDCT_FG'].T
TrainsampleDCT_BG = TrainingDCT['TrainsampleDCT_BG'].T
Z = np.loadtxt("test_resources/Zig-Zag Pattern.txt").flatten().astype(int)
FG_label = np.ones((TrainsampleDCT_FG.shape[1]))
BG_label = np.zeros((TrainsampleDCT_BG.shape[1]))

samples = np.hstack((TrainsampleDCT_FG, TrainsampleDCT_BG))
labels = np.hstack((FG_label, BG_label))

cheetah_classifier = Classifier(samples, labels)
cheetah_classifier.train()

test_image = plt.imread("test_resources/cheetah.bmp")
test_image = test_image.copy()
test_mask = plt.imread("test_resources/cheetah_mask.bmp") / 255
test_image = test_image[:, :-1, 0]
test_image[:, :-1] = test_image[:, :-1] - 60
test_image = test_image / 255
test_mask = test_mask[:test_mask.shape[0] - 8, :test_mask.shape[1] - 9]

cheetah_vectors = np.zeros((64, (test_image.shape[0] - 8) * (test_image.shape[1] - 8)))
cheetah_label = test_mask.flatten().astype(int)
n = 0
for i in range(test_image.shape[0] - 8):
    for j in range(test_image.shape[1] - 8):
        B = dct(dct(test_image[i:i + 8, j:j + 8].T, norm='ortho').T, norm='ortho')
        cheetah_vectors[Z, n] = B.flatten()
        n += 1

acc, predictions = cheetah_classifier.evaluate(cheetah_vectors, cheetah_label)
print(acc)
Pe = cheetah_classifier.prob_of_error(cheetah_label, predictions)
print(Pe)
