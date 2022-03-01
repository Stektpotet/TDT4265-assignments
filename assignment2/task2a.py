import warnings
from typing import Tuple

import numpy as np
from numpy import ndarray

import utils
import typing
np.random.seed(1)

def find_mean_std(X: np.ndarray) -> tuple[ndarray, ndarray]:
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        mean: the mean value of X
        std: the standard deviation of X
    """
    # print(np.alltrue((X.astype(float) / 255) == (X / 255)))  # Converting to float first has no significant effect
    X = X / 255  # scale to 0-1 range
    mean = np.mean(X)
    std = np.std(X)
    return mean, std

def pre_process_images(X: np.ndarray, mean: float = 0.13066, std: float = 0.3081078):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    X = ((X / 255) - mean) / std    # normalise
    X = np.append(X, np.ones((X.shape[0], 1), dtype=X.dtype), axis=1)  # add bias term
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return -(targets * np.log(outputs)).sum(axis=1).mean().item()


def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))

def deriv_sigmoid(Z: np.ndarray) -> np.ndarray:
    ez = np.exp(Z)
    return ez / (ez + 1) ** 2

def improved_sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1.7159 * np.tanh(2/3 * Z)

def deriv_improved_sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1.7259 * 2/3 * (1 - (np.tanh(2/3*Z))**2)

def softmax(Z: np.ndarray) -> np.ndarray:
    ez = np.exp(Z)
    return ez / np.sum(ez, axis=1, keepdims=True)


class SoftmaxModel:

    @staticmethod
    def _zeros_init(shape: tuple) -> np.ndarray:
        return np.zeros(shape, dtype=float)

    @staticmethod
    def _uniform_init(shape: tuple) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, shape)

    @staticmethod
    def _fanin_normal_init(shape: tuple) -> np.ndarray:
        return np.random.normal(0, 1/np.sqrt(shape[0]), shape)

    def count_params(self):
        return sum(np.prod(w.shape) for w in self.ws)

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785

        self.sigmoid = improved_sigmoid if use_improved_sigmoid else sigmoid
        self.deriv_sigmoid = deriv_improved_sigmoid if use_improved_sigmoid else deriv_sigmoid

        # self.softmax = softmax

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        self.hidden_a = [np.zeros(size, dtype=float) for size in self.neurons_per_layer[:-1]]
        self.hidden_z = [np.zeros(size, dtype=float) for size in self.neurons_per_layer[:-1]]

        weight_init = SoftmaxModel._fanin_normal_init if use_improved_weight_init else SoftmaxModel._uniform_init

        self.grads = []
        self.ws = []
        # Initialize the weights
        prev = self.I
        for i, size in enumerate(self.neurons_per_layer, start=1):
            w_shape = (prev, size)
            prev = size
            print(f"Initializing W_{i} to shape:", w_shape)
            self.ws.append(weight_init(w_shape))
            self.grads.append(np.zeros(w_shape, dtype=float))

        print("Total number of parameters:", self.count_params())

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        for layer, w in enumerate(self.ws[:-1]):
            self.hidden_z[layer] = (Z := np.dot(X, w))
            self.hidden_a[layer] = (X := self.sigmoid(Z))
        return softmax(np.dot(X, self.ws[-1]))

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        error = outputs - targets
        for layer in range(len(self.ws) - 1, 0, -1):
            self.grads[layer] = np.dot(self.hidden_a[layer - 1].T, error) / outputs.shape[0]
            error = np.dot(error, self.ws[layer].T) * self.deriv_sigmoid(self.hidden_z[layer - 1])

        self.grads[0] = np.dot(X.T, error) / outputs.shape[0]

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        for i in range(len(self.grads)):
            self.grads[i].fill(0)


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    return np.eye(num_classes, dtype=Y.dtype)[Y.reshape(-1)]


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()

    print("X_train mean: {0}, std: {1}".format(*find_mean_std(X_train)))

    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
