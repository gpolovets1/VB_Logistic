# Implementatino of Bayesian Multiclass Logistic Regression.
# Variational Bayes is applied for learning the parameters.
# ARD is used to regularize the classifier.

# References:
# 1) Jan Drugowitsch, Variational Bayesian inference for linear and logistic regression.
#    arXiv preprint arXiv: arXiv:1310.5438
# 2) Drugowitsch implementation: https://github.com/jdrugo/vb_logit/blob/master/vb_logit_fit_ard.m
# 3) Murphy, Kevin P. Machine Learning: A Probabilistic Perspective. Cambridge, MA: MIT, 2012.
#    Print. 234, 758-760

import numpy as np
import scipy.linalg
import scipy.special
import math
from data_preprocessor import DataPreprocessor
import random
from argparse import ArgumentParser
from scipy.stats import multivariate_normal


class MultiClassLogistic:
    def __init__(self, data_prep, iter=None, elbo_thresh=0.1, prune_thresh=1e-4,
                 max_iter=1000, a_0=1e-7, b_0=1e-8, verbose=True):
        """
        Accepts an data pre-processor which has formatted the data for ingestion
        by the classifier object. Hyperparameters settings may also be supplied
        to the classifier object.
        """
        self.data_prep = data_prep
        self.iter = iter
        self.elbo_thresh = elbo_thresh
        self.prune_thresh = prune_thresh
        self.max_iter = max_iter
        self.a_0 = a_0
        self.b_0 = b_0
        self.verbose = verbose
        # Extract the number of different classes in the label set.
        # Subtract one for identifiability.
        self.M = data_prep.enc.n_values_ - 1
        self.D = data_prep.D
        self.X = data_prep.X
        self.labels = data_prep.labels

    def lse(self, phi):
        return math.log(1 + sum(np.exp(phi_k) for phi_k in phi))

    def softmax(self, phi):
        return np.exp(phi)/np.exp(self.lse(phi))

    def lower_bound(self):
        """Calculate the ELBO value for a single iteration of VB."""

        E_p_y, E_p_w, E_p_a, E_q_w, E_q_a = 0, 0, 0, 0, 0

        ####### E_p_w
        E_p_w = - (self.D * self.M / 2) * np.log(2 * np.pi) - \
                0.5 * (np.trace(self.alpha.dot(self.V_N)) +
                       np.dot(np.dot(self.m_n.T, self.alpha.A), self.m_n).A1[0])

        alpha_n = self.a_0 + 0.5

        b_n = np.array([])
        for i in range(self.D * self.M):
            b_n = np.append(b_n, (np.square(self.m_n[i]) + np.diag(self.V_N)[i]))

        E_p_w += 0.5 * np.sum(scipy.special.psi(alpha_n) - np.log(b_n))

        ####### E_q_w
        sign, log_det_V_N = np.linalg.slogdet(self.V_N)
        E_q_w = -0.5 * sign * log_det_V_N - self.D / \
                                            2.0 * (1 + 2 * np.log(np.pi))

        for i in range(self.D * self.M):
            ####### E_p_a
            E_p_a += - np.log(scipy.special.gamma(self.a_0)) + \
                     self.a_0 * np.log(self.b_0) + \
                     (self.a_0 - 1) * (scipy.special.psi(alpha_n) -
                                       np.log(b_n[i])) - self.b_0 * (alpha_n / b_n[i])
            ####### E_q_a
            E_q_a += - np.log(scipy.special.gamma(alpha_n)) + \
                     (alpha_n - 1) * scipy.special.psi(alpha_n) + \
                     np.log(b_n[i]) - alpha_n

        for i in range(self.X.shape[0] / self.M):
            phi_i = self.phi[i* self.M: (i+1) * self.M]
            b_phi = np.dot(self.A, phi_i) - self.softmax(phi_i)

            c = 0.5 * np.dot(np.dot(np.transpose(phi_i), self.A), phi_i) - \
                np.dot(np.transpose(self.softmax(phi_i)), phi_i) + self.lse(phi_i)

            ####### E_p_y
            E_p_y += self.labels[i].dot(phi_i)[0] - 0.5 * np.dot(np.dot(phi_i.T, self.A), phi_i) + \
                     np.dot(b_phi.T, phi_i) - c

        return E_p_y + E_p_w + E_p_a + E_q_w + E_q_a

    def update_ARD_matrix(self):
        """
        Updates the matrix of alpha hyper-parameters which results in
        sparse posterior parameter values.
        """
        matrix_values = []
        for i in range(self.M):
            m_n_i = self.m_n[i * self.D: (i + 1) * self.D]
            V_N_i = np.diag(self.V_N)[i * self.D: (i + 1) * self.D]
            matrix_values.append(
                np.diag(np.array((self.a_0 + 0.5) /
                                 (0.5 * (np.square(m_n_i) + V_N_i.reshape(self.D, 1)) + self.b_0))
                        .flatten()))
        return scipy.sparse.block_diag(matrix_values)

    def initialize_ARD_matrix(self):
        matrix_values = []
        for i in range(self.M):
            matrix_values.append(np.diag(np.ones(self.D)) * float(self.a_0) / self.b_0)
        return scipy.sparse.block_diag(matrix_values)

    def optimize_eb(self):

        # Initialize hyperparameters
        self.bounds = []
        self.m_0 = np.zeros(self.M * self.D)

        # Set A matrix (Bohning upper bound on lse)
        self.A = 0.5 * (np.eye(self.M) - 1/(self.M + 1.0) *
                        np.outer(np.ones(self.M), np.ones(self.M)))

        self.m_n = self.m_0 # The prior is set to be 0. Initialize m_n to the prior.
        self.alpha = self.initialize_ARD_matrix() # alpha is the prior on the precision matrix

        # Run learning algorithm until convergence
        count = 0
        while True:
            self.phi = self.X.dot(self.m_n)
            # m & V are the posterior mean and variance respectively.
            temp_V_sum = None
            temp_m_sum = None
            for i in range(self.X.shape[0]/self.M):
                b_i = (np.dot(self.A, self.phi[i * self.M:(i +1) * self.M]) -
                       self.softmax(self.phi[i * self.M:(i +1) * self.M]))
                b_i = b_i.reshape(b_i.shape[0], 1)
                if temp_V_sum is not None:
                    temp_V_sum += np.dot(self.X[self.M * i:(i+1) * self.M].T.dot(self.A),
                                         self.X[self.M * i: (i+1)* self.M].A)
                else:
                    temp_V_sum = np.dot(self.X[self.M * i:(i+1) * self.M].T.dot(self.A),
                                        self.X[self.M * i: (i+1) * self.M].A)
                if temp_m_sum is not None:
                    temp_m_sum += np.transpose(self.X[self.M * i:(i + 1) * self.M])\
                        .dot(self.labels[i].T + b_i)
                else:
                    temp_m_sum = np.transpose(self.X[self.M * i:(i + 1) * self.M])\
                        .dot(self.labels[i].T + b_i)

            self.V_N = np.linalg.inv(self.alpha + temp_V_sum)
            self.m_n = np.dot(self.V_N, temp_m_sum)
            self.alpha = self.update_ARD_matrix()
            bound =  self.lower_bound()
            if self.verbose:
                print "Iteration = ", count
                print "bound = ", bound
            count += 1
            if self.iter is not None and self.iter == (count - 1):
                break
            if self.iter is None and (self.bounds and
                                     (abs((bound - self.bounds[-1])) < self.elbo_thresh
                                or bound == float("inf"))):
                break
            self.bounds.append(bound)

        # Set the posterior mean to 0 if it is small enough.
        self.m_n[1/np.diag(self.V_N) < self.prune_thresh] = 0

    def predict(self, test_data, test_labels):
        # The posterior predictive is now also a Gaussian.
        self.data_prep.create_X_matrix(test_data)
        self.data_prep.encode_labels(test_labels)
        self.X = self.data_prep.X
        self.labels = self.data_prep.labels
        predictions = []
        for i in range(self.X.shape[0]/self.M):
            predictions_i = []
            for j in range(self.M + 1):
                predict_mean = self.X[self.M * i:(i+1) * self.M] * self.m_n

                predict_var = self.X[self.M * i:(i+1) * self.M] * self.V_N * \
                              np.transpose(self.X[self.M * i:(i+1) * self.M]) + \
                              np.linalg.inv(self.A)
                # Transform y-hat back to y
                phi = self.X[self.M * i:(i+1) * self.M] * self.m_n

                b_phi = self.A * phi - self.softmax(phi)

                candidate_label = \
                    self.data_prep.enc.transform([[j]])[0] \
                        [:, 0:self.data_prep.enc.n_values_[0] - 1]

                y_hat = np.linalg.inv(self.A) * (b_phi + np.transpose(candidate_label))

                c = 0.5 * np.transpose(phi) * self.A * phi - \
                    np.transpose(self.softmax(phi)) * phi + self.lse(phi)

                h_phi = np.sqrt(np.linalg.det(2 * np.pi * np.linalg.inv(self.A))) * \
                        np.exp(0.5 * np.transpose(y_hat) * self.A * y_hat - c)

                posterior_prob = multivariate_normal(np.array(predict_mean).flatten(), predict_var)

                predictions_i.append(h_phi * posterior_prob.pdf(np.squeeze(np.array(y_hat))))

            predictions.append(np.argmax(predictions_i))
        return predictions


    def misclassification_rate(self, features, labels):
        predictions = self.predict(features, labels)
        assert len(labels) == len(predictions), \
            "You have %d # of labels and %d # of predictions!" % \
            (labels, predictions)
        return 1 - sum(labels == predictions)/float(len(labels))


def parse_command_line():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="Dataset to use. Options are 'iris', 'digits', 'synth'. ")
    parser.add_argument("--synth-x-dim", type=int, default=60,
                        help="Dimension of x if creating a synthetic dataset.")
    parser.add_argument("--synth-y-dim", type=int, default=3,
                        help="Dimension of y if creating a synthetic dataset.")
    parser.add_argument("--synth-x-dep", type=int, default=10,
                        help="The number of parameters in the synthetic dataset"
                             "that y is a function of (the rest would just be "
                             "noise variables).")
    parser.add_argument("--elbo-thresh", type=float, default=0.1,
                        help="The difference between successive ELBO values "
                             "required to stop the VB iterations.")
    parser.add_argument("--prune-threshold", type=float, default=1e-4,
                        help="The weight threhsold for which to remove parameters after "
                             "performing ARD regularization.")
    parser.add_argument("--iter", type=int,
                        help="The number of iteration to run VB for. Replaces"
                             "using the elbo threshold to stop optimizing.")
    parser.add_argument("--a0", type=float, default=1e-8,
                        help="Hyperparameter value for ARD. "
                             "Lower values produce maximal sparsity of parameters.")
    parser.add_argument("--b0", type=float, default=1e-8,
                        help="Hyperparameter value for ARD. "
                             "Lower values produce maximal sparsity of parameters.")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="The maximum allowable VB iterations.")
    parser.add_argument("-v", "--verbose", default=True,
                        help="Whether to print intermediate information during VB "
                             "optimization process.")
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(10)
    random.seed(10)
    args = parse_command_line()
    data_prep = DataPreprocessor()
    inputs, targets, dep_indices, dep_weights = \
        data_prep.load_dataset(args.dataset, args.synth_x_dim,
                               args.synth_x_dep, args.synth_y_dim)
    train_index = random.sample(range(0, inputs.shape[0]), 2 * inputs.shape[0] / 3)
    test_index = list(set(range(inputs.shape[0])).difference(train_index))
    train_inputs = inputs[train_index, :]
    train_targets = targets[train_index,]
    data_prep = DataPreprocessor(train_inputs, train_targets)
    print "loaded input_gen"
    data_prep.encode_labels()
    print "encoded y labels"
    data_prep.create_X_matrix()
    print "created X matrix"
    mc_log = MultiClassLogistic(data_prep, args.iter, args.elbo_thresh, args.prune_threshold,
                                args.max_iter, args.a0, args.b0, args.verbose)
    mc_log.optimize_eb()
    print mc_log.misclassification_rate(inputs[test_index, :], targets[test_index])









