import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode

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
    assert pred.shape == targets.shape, \
        f"Model prediction shape: {pred.shape}, targets: {targets.shape}"
    return (pred.argmax(axis=1) == targets.argmax(axis=1)).mean().item()


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray) -> float:
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

        pred = self.model.forward(X_batch)
        loss = cross_entropy_loss(Y_batch, pred)
        self.model.zero_grad()
        self.model.backward(X_batch, pred, Y_batch)
        self.model.w = self.model.w - self.learning_rate * self.model.grad  # Gradient descent step
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
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val



if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png", dpi=300)
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .95])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png", dpi=300)
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model_l2 = SoftmaxModel(l2_reg_lambda=2.0)
    trainer = SoftmaxTrainer(
        model_l2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    print("(L2)Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model_l2.forward(X_train)))
    print("(L2)Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model_l2.forward(X_val)))
    print("(L2)Final Train accuracy:", calculate_accuracy(X_train, Y_train, model_l2))
    print("(L2)Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model_l2))


    l0_img = np.concatenate([model.w[:-1, i].reshape(28, 28) for i in range(10)], axis=1)
    l2_img = np.concatenate([model_l2.w[:-1, i].reshape(28, 28) for i in range(10)], axis=1)

    fig = plt.figure(figsize=(28, 5.6))
    ax_l0 = fig.add_axes([0, 0.5, 1, 0.5])
    ax_l0.set_axis_off()
    # ax_l0.set_title("L2 Regularization ($\\lambda$ = 0.0)", fontsize=16)
    ax_l0.imshow(l0_img, cmap="gray")
    ax_l2 = fig.add_axes([0, 0, 1, 0.5])
    ax_l2.set_axis_off()
    # ax_l2.set_title("L2 Regularization ($\\lambda$ = 2.0)", fontsize=16)
    ax_l2.imshow(l2_img, cmap="gray")
    plt.savefig("task4b_softmax_l2_regularization.png", dpi=300)
    plt.show()
    # plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [2, .2, .02, .002]
    weight_images = []
    weight_norms = []
    for h in l2_lambdas:
        trainer = SoftmaxTrainer(
            SoftmaxModel(l2_reg_lambda=h), learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val
        )
        train_history_reg, val_history_reg = trainer.train(num_epochs)
        # Plot accuracy
        utils.plot_loss(val_history_reg["accuracy"], f"Validation Accuracy ($\\lambda={h}$)")
        weight_norms.append(np.linalg.norm(trainer.model.w))
        weight_images.append(np.concatenate([trainer.model.w[:-1, i].reshape(28, 28) for i in range(10)], axis=1))

    plt.ylim([0.68, .95])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png", dpi=300)
    plt.show()

    fig = plt.figure(figsize=(28, 11.2))
    for i, wimg in enumerate(weight_images):
        ax = fig.add_axes([0, i * 0.25, 1, 0.25])
        ax.set_axis_off()
        ax.imshow(wimg, cmap="gray")
    plt.savefig("task4b_softmax_l2_regularization_full.png", dpi=300)
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight

    plt.plot(l2_lambdas, weight_norms)
    plt.scatter(l2_lambdas, weight_norms)
    for x, y, offset in zip(l2_lambdas, weight_norms, [(-35, 30), *((5, -4) for _ in range(len(l2_lambdas)-1))]):
        plt.annotate(f'({x}, {y:.3f})', (x, y), textcoords='offset points', xytext=offset)
    plt.xscale("log")
    plt.xlabel("L2 $\\lambda$")
    plt.ylabel("L2 Norm of Weights")
    plt.savefig("task4d_l2_reg_norms.png", dpi=300)
    plt.show()