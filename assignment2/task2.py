import numpy
import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
from trainer import BaseTrainer
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    pred = model.forward(X)
    accuracy = np.mean(pred.argmax(axis=1) == targets.argmax(axis=1))
    return accuracy.item()


class SoftmaxTrainer(BaseTrainer):

    def _weight_update(self):
        for i, (w, grads) in enumerate(zip(self.model.ws, self.model.grads)):
            self.model.ws[i] = w - self.learning_rate * grads

    def _weight_update_momentum(self):
        for i, (w, grad, last_grad) in enumerate(zip(self.model.ws, self.model.grads, self._previous_grads)):
            self._previous_grads[i] = grad + self._momentum_gamma * last_grad     # update what previous grad will be
            self.model.ws[i] = w - self.learning_rate * self._previous_grads[i]  # update ws using the updated grad

    def __init__(
            self,
            momentum_gamma: float,
            use_momentum: bool,
            learning_rate_schedule: dict[int, float] = None,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._step = 0
        self.schedule = learning_rate_schedule if learning_rate_schedule is not None else {}
        self._previous_grads = [np.zeros_like(g, g.dtype) for g in self.model.grads] if use_momentum else None
        self._momentum_gamma = momentum_gamma
        self._weight_update = self._weight_update_momentum if use_momentum else self._weight_update


    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        if self._step in self.schedule:
            self.learning_rate = self.schedule[self._step]
            print("Step {}: Learning rate changed to {}".format(self._step, self.learning_rate))

        pred = self.model.forward(X_batch)
        loss = cross_entropy_loss(Y_batch, pred)
        self.model.zero_grad()
        self.model.backward(X_batch, pred, Y_batch)
        self._weight_update()

        self._step += 1
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Settings for task 3. Keep all to false for task 2.
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist(train_size=60000, val_size=10000, sample_stochastic=True)
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    # Hyperparameters

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum, None,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs, use_early_stopping=False)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Plot loss for first model (task 2c)
    _ = plt.figure(figsize=(10, 5))
    plt.ylim([0., .5])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task2c_losses.png")
    plt.show()

    _ = plt.figure(figsize=(10, 5))
    # Plot accuracy
    plt.gcf().clear()
    plt.ylim([0.90, 1])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task2c_accuracies.png")
    plt.show()
