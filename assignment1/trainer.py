import numpy as np
import utils


class BaseTrainer:

    def __init__(
            self,
            model,
            learning_rate: float,
            batch_size: int,
            shuffle_dataset: bool,
            X_train: np.ndarray, Y_train: np.ndarray,
            X_val: np.ndarray, Y_val: np.ndarray,) -> None:
        """
            Initialize the trainer responsible for performing the gradient descent loop.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model
        self.shuffle_dataset = shuffle_dataset

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
        pass

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray) -> float:
        """
            Perform forward, backward and gradient descent step here.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        pass

    def train(self, num_epochs: int, use_early_stopping: bool = False):
        """
        Training loop for model.
        Implements stochastic gradient descent with num_epochs passes over the train dataset.
        Returns:
            train_history: a dictionary containing loss and accuracy over all training steps
            val_history: a dictionary containing loss and accuracy over a selected set of steps
        """
        # Utility variables
        num_batches_per_epoch = self.X_train.shape[0] // self.batch_size
        num_steps_per_val = 1 #num_batches_per_epoch // 5
        # A tracking value of loss over all training steps
        train_history = dict(
            loss={},
            accuracy={}
        )
        val_history = dict(
            loss={},
            accuracy={}
        )

        global_step = 0
        for epoch in range(num_epochs):
            train_loader = utils.batch_loader(
                self.X_train, self.Y_train, self.batch_size, shuffle=self.shuffle_dataset)
            for X_batch, Y_batch in iter(train_loader):
                loss = self.train_step(X_batch, Y_batch)
                # Track training loss continuously
                train_history["loss"][global_step] = loss

                # Track validation loss / accuracy every time we progress 20% through the dataset
                if global_step % num_steps_per_val == 0:
                    val_loss, accuracy_train, accuracy_val = self.validation_step()
                    train_history["accuracy"][global_step] = accuracy_train
                    val_history["loss"][global_step] = val_loss
                    val_history["accuracy"][global_step] = accuracy_val

                    # ============================    EARLY STOPPING    ============================
                    if use_early_stopping and len(val_history["loss"]) > 10:
                        for vl in (val_history["loss"][global_step - num_steps_per_val*i] for i in range(1, 10+1)):
                            if val_loss < vl:
                                break  # Validation loss has improved within the last 10 steps, continue training
                        else:  # Otherwise, early stopping
                            print("Early stopping!\n"
                                  f"Early stop point reached at step {global_step} due to insufficient improvement.")
                            return train_history, val_history
                    # ============================  END EARLY STOPPING  ============================

                global_step += 1
        return train_history, val_history