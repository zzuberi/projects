import numpy as np
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm


class Classifier:
    """
    This class creates a Gaussian BDR classifier from given training data and corresponding labels.

    Gaussian parameters are learned using Maximum Likelihood estimation using the training data.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
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

        self.mean = np.array(self.mean)
        self.variance = np.array(self.variance)
        self.priors = np.array(self.priors)
        print()

    def evaluate(self, data: np.ndarray, labels: np.ndarray, dims=None) -> (float, np.ndarray):
        """
        Evaluate model on given test data and corresponding labels.
        :param data: dxn array containing n examples of dimensionality d
        :param labels: n array containing integer labels corresponding to each example in data array
        :param dims: n array containing indexes for which dimensions to evaluate data on. Data will only be evaluated on
        the specified dimensions usin the marginal distributions of those dimensions
        :returns accuracy: model accuracy on given test data
                 predicted_labels: n array containing predicted integer labels corresponding to each example in test
                 data array
        """
        assert (self.classes is not None or self.c is not None or self.n is not None or self.mean is not None or
                self.variance is not None or self.priors is not None), "Model must be trained first."
        print("Evaluating Model...")
        print()

        mean, variance = self.get_marginal_distributions(dims=dims)
        n = data.shape[1]
        post_prob = np.zeros((self.c, n))
        if dims is not None:
            data = data[dims, :].copy()
        for cls in tqdm(self.classes):
            post_prob[cls, :] = self.priors[cls] * mvn.pdf(data.T, mean[cls], variance[cls])

        predicted_labels: np.ndarray = post_prob.argmax(axis=0)

        accuracy = sum(predicted_labels == labels) / n
        print()
        return accuracy, predicted_labels

    def prob_of_error(self, gt: np.ndarray, pred: np.ndarray) -> float:
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

    def get_marginal_distributions(self, cls=None, dims=None):
        if cls is None and dims is None:
            return self.mean, self.variance

        if cls is None:
            cls = self.classes

        if dims is None:
            dims = range(0, self.data.shape[0])

        means = self.mean[tuple(np.meshgrid(cls, dims, indexing='ij'))]
        variance = self.variance[tuple(np.meshgrid(cls, dims, dims, indexing='ij'))]

        return means, variance
