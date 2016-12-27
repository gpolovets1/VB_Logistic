import scipy
import numpy as np
from sklearn import datasets, preprocessing
from scipy.sparse import csr_matrix

class DataPreprocessor:

    def __init__(self, inputs=None, labels=None):
        self.labels = labels
        self.input_data = inputs

    def load_dataset(self, datsaset_name, synth_x_dim=None, synth_x_dep=None, synth_y_dim=None):
        if datsaset_name == "iris":
            dataset = datasets.load_iris()
            return dataset.data, dataset.target, None, None
        elif datsaset_name == "digits":
            dataset = datasets.load_digits()
            return dataset.data, dataset.target, None, None
        elif datsaset_name == "synth":
            return self.create_ard_data(xdim=synth_x_dim,
                                        x_dep=synth_x_dep,
                                        y_dim=synth_y_dim)

    def create_ard_data(self, xdim=60, x_dep=10, y_dim=3):
        """
        Creates a synthetic dataset where the response only depends on a subset
        of input variables.
        :param xdim: The dimension of the input.
        :param x_dep: The number of input variables that the response depends on.
        :param y_dim: The number of classes in the response.
        :return:
        """
        np.random.seed(0)
        x = [[np.random.normal(0, 1) for _ in range(xdim)] for __ in range(1000)]
        x_dep_indices = [np.random.randint(xdim) for _ in range(x_dep)]
        dep_weights = [[np.random.normal(0, 1)
                        for _ in range(x_dep)] for __ in range(y_dim)]

        y = [np.argmax(DataPreprocessor._class_linear_sums(
            x[i], dep_weights, x_dep_indices)) for i in range(1000)]
        return np.array(x), np.array(y), x_dep_indices, dep_weights

    @staticmethod
    def _class_linear_sums(x_values, weights, x_indices=None):
        """ Produces the sum of the products between the input variables and
        weights."""
        linear_sums = []
        for i in range(len(weights)):
            if x_indices:
                linear_sums.append(
                    np.sum(np.multiply(
                        [x_values[j] for j in x_indices], weights[i])))
            else:
                linear_sums.append(np.sum(np.multiply(x_values, weights[i])))
        return linear_sums

    def encode_labels(self, new_labels=None):
        # Stores the labels as a one-hot encoded sparse matrix
        # (a one-hot encoded vector for each label datum).
        self.enc = preprocessing.OneHotEncoder()
        labels = new_labels if new_labels is not None else self.labels
        label_set = set(labels)
        self.enc.fit([[label] for label in list(sorted(label_set))])
        encoded_labels = None
        for i, label in enumerate(labels):
            # Exclude last label for idenfiability
            encoded_labels = scipy.sparse.vstack([encoded_labels,
                                               self.enc.transform([[label]])[0]
                                               [:, 0:self.enc.n_values_[0] - 1]]) \
                if encoded_labels is not None else \
                self.enc.transform([[label]])[0][:, 0:self.enc.n_values_[0] - 1]
        self.labels = encoded_labels

    def create_X_matrix(self, input_data=None):
        # Generates a sparse matrix where each input datum of length D
        # is replicated for M rows (where M is the number of classes).
        # Each replicated row i is sparse and is of length M*D.
        # The only nonzero values of row i are between the indices of
        # [i*D, (i+1) * D] which will be used to calculate the parameters
        # for class i.
        self.X = None
        input_data = input_data if input_data is not None else self.input_data
        self.D = len(input_data[0])
        for i in range(len(input_data)):
            self.X = scipy.sparse.vstack(
                [self.X, self.create_X_row(input_data[i])]) \
                if self.X is not None else \
                self.create_X_row(input_data[i])
        self.X = csr_matrix(self.X)

    def create_X_row(self, single_row):
        matrix_values = []
        for i in range(self.enc.n_values_ - 1):
            matrix_values.append(csr_matrix(single_row))
        return scipy.sparse.block_diag(matrix_values)