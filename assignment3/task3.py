import os.path
import pathlib
import pickle
import typing

import matplotlib.pyplot as plt
import torch

import utils
from torch import nn
from dataloaders import load_cifar10
from task2 import ExampleModel
from trainer import Trainer
from torchvision import transforms as T


class ResidualSkipCat(nn.Sequential):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, super().forward(x)), dim=1)

class PrintModule(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x

class CNNA(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes):
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
            nn.Conv2d(in_channels=image_channels, out_channels=num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            # 32 x 16 x 16
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters * 2), nn.ReLU(inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters * 4), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            # 128 x 8 x 8
            ResidualSkipCat(*[nn.Sequential(
                nn.Conv2d(num_filters * 4, num_filters * 4, 3, 1, 1),
                nn.BatchNorm2d(num_filters * 4), nn.PReLU(), nn.Dropout2d(p=0.2))
                for _ in range(2)
            ]),
            # 256 x 8 x 8
            ResidualSkipCat(
                nn.Conv2d(num_filters * 4 * 2, num_filters * 4, 3, 1, 1),
                nn.BatchNorm2d(num_filters * 4), nn.PReLU(), nn.Dropout2d(p=0.2),
                nn.Conv2d(num_filters * 4, num_filters * 4, 3, 1, 1),
                nn.BatchNorm2d(num_filters * 4), nn.PReLU(), nn.Dropout2d(p=0.2)
            ),
            # 384 x 8 x 8
            ResidualSkipCat(
                nn.Conv2d(num_filters * 4 * 3, num_filters * 4, 3, 1, 1),
                nn.BatchNorm2d(num_filters * 4), nn.PReLU(), nn.Dropout2d(p=0.2),
                nn.Conv2d(num_filters * 4, num_filters * 4, 3, 1, 1),
                nn.BatchNorm2d(num_filters * 4), nn.PReLU(), nn.Dropout2d(p=0.2)
            ),
            # 512 x 8 x 8
            nn.Conv2d(in_channels=num_filters * 4 * 4, out_channels=num_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters * 4), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
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



class TrainerV2(Trainer):

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader]):
        super().__init__(batch_size, learning_rate, early_stop_count, epochs, model, dataloaders)

        # Override parameters set in Trainer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, amsgrad=True)
        self.checkpoint_dir = pathlib.Path("checkpoints_task3")


def create_plots(trainer: Trainer, name: str):
    final_evaluation_metrics = trainer.full_evaluation()
    last_key = next(reversed(trainer.train_history["loss"]))

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
    batch_size = 256
    learning_rate = 5e-4
    early_stop_count = 4

    augmentation = [
        # T.RandomHorizontalFlip(),
        # T.RandomAutocontrast(),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),
        # T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ]

    dataloaders = load_cifar10(batch_size, augmentation=augmentation)
    trainer1 = TrainerV2(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 16 x 16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 32 x 8 x 8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64 x 4 x 4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 128 x 2 x 2
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 256 x 1 x 1
            nn.Flatten(),
            nn.Linear(256, 10)
        ),
        dataloaders
    )
    trainer2 = TrainerV2(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        CNNA(image_channels=3, num_classes=10),
        dataloaders
    )
    trainer1.train()
    create_plots(trainer1, "task3_augmented")
    trainer2.train()
    create_plots(trainer2, "task3_model_refined")
    #
    # load_model = True
    #
    # if os.path.exists("checkpoints/latest_checkpoint") and \
    #         os.path.exists("./task2_history.pkl") and load_model:
    #     with open("checkpoints/latest_checkpoint", "r") as f:
    #         latest_checkpoint = f.read()
    #     state_dict = torch.load(os.path.join("checkpoints", latest_checkpoint))
    #     trainer.model.load_state_dict(state_dict)
    #     print(f"Loaded model from latest checkpoint: {latest_checkpoint}")
    #     history = pickle.load(open("task2_history.pkl", "rb"))
    #     trainer.train_history = history["train"]
    #     trainer.validation_history = history["validation"]
    # else:
    #     trainer.train()
    #     history = {'train': trainer.train_history, 'validation': trainer.validation_history}
    #     pickle.dump(history, open("task2_history.pkl", "wb"))


if __name__ == "__main__":
    main()