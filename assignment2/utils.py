from typing import Generator
import mnist
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def batch_loader(
        X: np.ndarray, Y: np.ndarray,
        batch_size: int, shuffle=False,
        drop_last=True) -> Generator:
    """
    Creates a batch generator over the whole dataset (X, Y) which returns a generator iterating over all the batches.
    This function is called once each epoch.
    Often drop_last is set to True for the train dataset, but not for the train set.

    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        Y: labels of shape [batch size]
        drop_last: Drop last batch if len(X) is not divisible by batch size
        shuffle (bool): To shuffle the dataset between each epoch or not.
    """
    assert len(X) == len(Y)
    num_batches = len(X) // batch_size
    if not drop_last:
        num_batches = int(np.ceil(len(X) / batch_size))

    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(num_batches):
        # select a set of indices for each batch of samples
        batch_indices = indices[i*batch_size:(i+1)*batch_size]
        x = X[batch_indices]
        y = Y[batch_indices]
        # return both images (x) and labels (y)
        yield (x, y)


### NO NEED TO EDIT ANY CODE BELOW THIS ###

def load_full_mnist(train_size: int = 20000, val_size: int = 10000, sample_stochastic: bool = True):
    """
    Loads and splits the dataset into train, validation and test.
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        sample_stochastic: If True, the subset is sampled stochastically.

    Returns:
        X_train: images of shape [train_size, 784] in the range (0, 255)
        Y_train: labels of shape [train_size]
        X_val: images of shape [test_size, 784] in the range (0, 255)
        Y_val: labels of shape [test_size]
    """
    X_train, Y_train, X_val, Y_val = mnist.load()

    train_size = train_size if train_size > 0 else len(X_train)
    val_size = val_size if val_size > 0 else len(X_val)

    if train_size > X_train.shape[0]:
        raise ValueError(f"train_size ({train_size}) is greater than the number of training samples ({X_train.shape[0]})")
    if val_size > X_val.shape[0]:
        raise ValueError(f"val_size ({val_size}) is greater than the number of validation samples ({X_val.shape[0]})")

    if sample_stochastic:
        train_idx = np.random.choice(X_train.shape[0], train_size, replace=False)
        val_idx = np.random.choice(X_val.shape[0], val_size, replace=False)
    else:
        # Default to first 'train_size' of train set images for training
        # and last 'test_size' from test set images for validation
        train_idx = np.arange(train_size)
        val_idx = np.arange(X_val.shape[0] - val_size, X_val.shape[0])

    # Subset sampling
    X_train, Y_train = X_train[train_idx], Y_train[train_idx]
    X_val, Y_val = X_val[val_idx], Y_val[val_idx]

    # Reshape to (N, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)

    print(f"Train shape: X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Validation shape: X: {X_val.shape}, Y: {Y_val.shape}")

    return X_train, Y_train, X_val, Y_val


def plot_loss(loss_dict: dict, label: str = None, npoints_to_average=1, plot_variance=True):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
        npoints_to_average: Number of points to average plot
    """
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    if npoints_to_average == 1:
        plt.plot(global_steps, loss, label=label)
        return

    num_points = len(loss) // npoints_to_average
    mean_loss = []
    loss_std = []
    steps = []
    for i in range(num_points):
        points = loss[i*npoints_to_average:(i+1)*npoints_to_average]
        step = global_steps[i*npoints_to_average + npoints_to_average//2]
        mean_loss.append(np.mean(points))
        loss_std.append(np.std(points))
        steps.append(step)
    plt.plot(steps, mean_loss,
             label=f"{label} (mean over {npoints_to_average} steps)")
    if plot_variance:
        plt.fill_between(
            steps, np.array(mean_loss) -
            np.array(loss_std), np.array(mean_loss) + loss_std,
            alpha=.2, label=f"{label} variance over {npoints_to_average} steps")
