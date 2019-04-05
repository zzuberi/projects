import numpy as np
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm


class Classifier:
    """
    This class creates a Gaussian BDR classifier from given training data and corresponding labels.

    Gaussian parameters are learned using Maximum Likelihood estimation using the training data.
    """

    def __init__(self, data, labels):
        """
        Constructor for Classifier class.
        :param data: dxn array containing n examples of dimensionality d
        :param labels: n array containing integer labels corresponding to each example in data array
        """
        self.data = data
        self.labels = np.squeeze(labels)
        self.classes = None
        self.c = None
        self.n = None
        self.mean = None
        self.variance = None
        self.priors = None
        return

    def train(self):
        """
        Method determines the parameters for the classifier using Maximum Likelihood estimation.
        """
        print('Training Model...')
        print()
        self.classes = np.unique(self.labels).astype(int)
        self.c = self.classes.shape[0]
        self.n = self.data.shape[1]
        self.mean = []
        self.variance = []
        self.priors = []

        for cls in tqdm(self.classes):
            class_examples = np.where(self.labels == cls)[0]
            examples = self.data[:, class_examples]
            self.mean.append(examples.mean(axis=1))
            self.variance.append(np.cov(examples))
            self.priors.append(class_examples.shape[0] / self.labels.shape[0])
        print()

    def evaluate(self, data, labels):
        """
        Evaluate model on given test data and corresponding labels.
        :param data: dxn array containing n examples of dimensionality d
        :param labels: n array containing integer labels corresponding to each example in data array
        :returns accuracy: model accuracy on given test data
                 predicted_labels: n array containing predicted integer labels corresponding to each example in test
                 data array
        """
        assert (self.classes is not None or self.c is not None or self.n is not None or self.mean is not None or \
                self.variance is not None or self.priors is not None), "Model must be trained first."
        print("Evaluating Model...")
        print()
        n = data.shape[1]
        post_prob = np.zeros((self.c, n))
        for cls in tqdm(self.classes):
            post_prob[cls, :] = self.priors[cls] * mvn.pdf(data.T, self.mean[cls], self.variance[cls])

        predicted_labels = post_prob.argmax(axis=0)

        accuracy = sum(predicted_labels == labels) / n
        print()
        return accuracy, predicted_labels

    def prob_of_error(self, gt, pred):
        """
        Evaulates produces the probability of error given ground truth and predicted labels.
        :param gt: n array containing groundtruth labels
        :param pred: n array containing predicted labels
        :return: probability of error of trained model
        """
        pe = []
        for cls in self.classes:
            in_class = np.where(gt == cls)[0]
            prior = in_class.size / gt.shape[0]
            out_class = np.where(pred[in_class] != cls)[0]
            pe.append(prior * out_class.size / in_class.size)
        return sum(pe)
