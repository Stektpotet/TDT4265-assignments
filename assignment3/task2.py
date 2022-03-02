import os.path
import pathlib
import pickle

import matplotlib.pyplot as plt
import torch

import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer


class ExampleModel(nn.Module):

    def __init__(self, image_channels: int, num_classes: int, kernel_size: int = 5, padding: int = 2):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(image_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            # 32 x 16 x 16
            nn.Conv2d(num_filters, num_filters * 2,  kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            # 64 x 8 x 8
            nn.Conv2d(num_filters * 2, num_filters * 4,  kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            # 128 x 4 x 4
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]

        self.num_output_features = (num_filters * 4) * 4 * 4 # 128 x 4 x 4

        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),  # Softmax applied after this
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]

        Returns:
            Logits from the final fully connected layer, shape: [batch_size, self.num_classes]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = self.classifier(x.view(batch_size, -1))
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    final_evaluation_metrics = trainer.full_evaluation()
    last_key = next(reversed(trainer.validation_history["loss"]))

    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.title("Cross Entropy Loss")
    plt.ylim(.0, 2.0)
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Accuracy")
    plt.ylim(.4, .9)
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.hlines(final_evaluation_metrics['train'][1], 0, last_key, colors='r', linestyles='dashed',
               label=f'Final Train Accuracy = {final_evaluation_metrics["train"][1]:.4f}')
    plt.hlines(final_evaluation_metrics['val'][1], 0, last_key, colors='g', linestyles=(0, (2, 2)),
               label=f'Final Validation Accuracy = {final_evaluation_metrics["val"][1]:.4f}')
    plt.hlines(final_evaluation_metrics['test'][1], 0, last_key, colors='b', linestyles='dashdot',
               label=f'Final Test Accuracy = {final_evaluation_metrics["test"][1]:.4f}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64         # 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    load_model = True

    if os.path.exists("checkpoints/latest_checkpoint") and \
            os.path.exists("./task2_history.pkl") and load_model:
        with open("checkpoints/latest_checkpoint", "r") as f:
            latest_checkpoint = f.read()
        state_dict = torch.load(os.path.join("checkpoints", latest_checkpoint))
        trainer.model.load_state_dict(state_dict)
        print(f"Loaded model from latest checkpoint: {latest_checkpoint}")
        history = pickle.load(open("task2_history.pkl", "rb"))
        trainer.train_history = history["train"]
        trainer.validation_history = history["validation"]
    else:
        trainer.train()
        history = {'train': trainer.train_history, 'validation': trainer.validation_history}
        pickle.dump(history, open("task2_history.pkl", "wb"))

    create_plots(trainer, "task2")

if __name__ == "__main__":
    main()