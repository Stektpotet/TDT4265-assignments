import os.path
import pickle

import matplotlib.transforms

import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    load_percent = 100

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist(600 * load_percent, 100 * load_percent, sample_stochastic=True)
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # For ease visualisation, we store aside the history of the training,
    # letting us change around plotting parameters later without having to re-train the models
    try_load_history = True
    history_dumpfile = f"task3_{load_percent}pct_history.pkl"

    all_history = dict()
    if try_load_history and os.path.exists(history_dumpfile):
        print("Loading history from file...")
        all_history = pickle.load(open(history_dumpfile, 'rb'))
    all_trainers = {
        'standard': SoftmaxTrainer(
            momentum_gamma, False, None,
            SoftmaxModel(neurons_per_layer, False, False),
            learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'winit': SoftmaxTrainer(
            momentum_gamma, False, None,
            SoftmaxModel(neurons_per_layer, False, True),
            learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'isig': SoftmaxTrainer(
            momentum_gamma, False, None,
            SoftmaxModel(neurons_per_layer, True, True),
            learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'momentum': SoftmaxTrainer(
            momentum_gamma, True, None,
            SoftmaxModel(neurons_per_layer, True, True),
            0.01, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'standard_lrscheduled': SoftmaxTrainer(
            momentum_gamma, False, {2e4: 2e-2, 60000: 2e-3, 80000: 2e-4},
            SoftmaxModel(neurons_per_layer, False, False),
            learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'winit_lrscheduled': SoftmaxTrainer(
            momentum_gamma, False, {2e4: 2e-2, 60000: 2e-3, 80000: 2e-4},
            SoftmaxModel(neurons_per_layer, False, True),
            learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'isig_lrscheduled': SoftmaxTrainer(
            momentum_gamma, False, {2e4: 2e-2, 60000: 2e-3, 80000: 2e-4},
            SoftmaxModel(neurons_per_layer, True, True),
            learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'momentum_lrscheduled': SoftmaxTrainer(
            momentum_gamma, True, {2e4: 2e-3, 60000: 2e-4, 80000: 2e-5},
            SoftmaxModel(neurons_per_layer, True, True),
            0.01, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'isig_solo': SoftmaxTrainer(
            momentum_gamma, False, None,
            SoftmaxModel(neurons_per_layer, True, False),
            learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'momentum_solo': SoftmaxTrainer(
            momentum_gamma, True, None,
            SoftmaxModel(neurons_per_layer, False, False),
            0.01, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'winit_momentum': SoftmaxTrainer(
            momentum_gamma, True, None,
            SoftmaxModel(neurons_per_layer, False, True),
            0.01, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
        'isig_momentum': SoftmaxTrainer(
            momentum_gamma, True, None,
            SoftmaxModel(neurons_per_layer, True, False),
            0.01, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        ),
    }

    all_trainers = {k: all_trainers[k] for k in all_trainers if k not in all_history}
    if len(all_trainers) > 0:
        for name, trainer in all_trainers.items():
            print(f"Training {name}...")
            train_history, val_history = trainer.train(num_epochs)
            all_history[name] = {'train': train_history, 'val': val_history}

        pickle.dump(all_history, open(history_dumpfile, "wb"))

    run_names = { # The naming tags are a little nonsensical, to refrain from having to re-train all models
        "standard": "No additions",
        "winit": "Weight Init",
        "isig": "Weight Init & Better Sigmoid",
        "momentum": "All additions [$\\alpha = 0.01,\\, \\gamma = 0.9$]",
        'isig_solo': "Better Sigmoid",
        'momentum_solo': "Momentum [$\\alpha = 0.01,\\, \\gamma = 0.9$]",
        'winit_momentum': "Weight Init & Momentum [$\\alpha = 0.01,\\, \\gamma = 0.9$]",
        'isig_momentum': "Better Sigmoid & Momentum [$\\alpha = 0.01,\\, \\gamma = 0.9$]",
        'standard_lrscheduled': " LR Schedule [2e4: 2e-2, 6e4: 2e-3, 8e4: 2e-4]",
        'winit_lrscheduled': "Weight Init & LR Schedule [2e4: 2e-2, 6e4: 2e-3, 8e4: 2e-4]",
        'isig_lrscheduled': "Weight Init & Better Sigmoid & LR Schedule [2e4: 2e-2, 6e4: 2e-3, 8e4: 2e-4]",
        'momentum_lrscheduled': "All additions & LR Schedule [$\\alpha = 0.01,\\, \\gamma = 0.9$ | 2e4: 2e-3, 6e4: 2e-4, 8e4: 2e-5]",
    }

    ylim_ranges = {name: {'loss': (0., .35), 'accuracy': (.9, 1.0025)} for name in all_history.keys()}
    ylim_ranges['standard']['loss'] = (0., .5)

    if not os.path.exists('./task3'):
        os.mkdir('task3')

    all_history = {k: v for k, v in all_history.items() if k in run_names}
    # Plotting
    for name_id, history in all_history.items():
        # ================================ PLOT LOSS ================================
        colors = plt.rcParams["axes.prop_cycle"]()
        next(colors)
        next(colors)
        run_name = run_names[name_id]
        train_history, val_history = history['train'], history['val']
        last_key = max(val_history['loss'].keys())
        _ = plt.figure(figsize=(10, 5))
        plt.ylim(ylim_ranges[name_id]['loss'])
        utils.plot_loss(train_history['loss'], f"Training Loss", npoints_to_average=25)
        utils.plot_loss(val_history['loss'], f"Validation Loss")
        min_key = min(val_history['loss'], key=val_history['loss'].get)
        loss_min = val_history['loss'][min_key]
        final_loss = val_history['loss'][last_key]
        plt.axhline(y=loss_min, c=next(colors)["color"], linestyle="--",
                    label="Validation Loss Minimum = {:.4f}".format(loss_min))
        plt.axhline(y=final_loss, c=next(colors)["color"], linestyle="--",
                    label="Final Validation Loss = {:.4f}".format(final_loss))

        plt.plot(min_key, loss_min, 'o', c=next(colors)['color'], label=f"Minimum Validation Loss, step {min_key}")
        plt.annotate("",
                    xy=(last_key, final_loss), xycoords='data',
                    xytext=(last_key, loss_min), textcoords='data',
                    arrowprops=dict(arrowstyle='<|-|>',
                                    shrinkA=0.0,
                                    shrinkB=0.0,
                                    connectionstyle="arc3"),
                    )
        text_offset = matplotlib.transforms.offset_copy(plt.gca().transData, plt.gcf(), x=16, y=-6, units='points')
        plt.text(last_key, loss_min, f"$final - minimum={final_loss - loss_min:.4f}$",
                 va='top', ha='right', transform=text_offset)

        plt.xlabel("Number of Training Steps")
        plt.ylabel("Cross Entropy Loss - Average")
        plt.title(f"Training and Validation Loss ({run_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"task3/task3_losses_{name_id}.png")
        plt.show()

        # ============================== PLOT ACCURACY ==============================
        colors = plt.rcParams["axes.prop_cycle"]()
        next(colors)
        next(colors)
        _ = plt.figure(figsize=(10, 5))
        plt.ylim(ylim_ranges[name_id]['accuracy'])

        utils.plot_loss(train_history["accuracy"], f"Training Accuracy ({run_name})")
        utils.plot_loss(val_history["accuracy"], f"Validation Accuracy ({run_name})")

        train_maximum = max(train_history["accuracy"].values())
        val_maximum = max(val_history["accuracy"].values())
        plt.axhline(y=train_maximum, c=next(colors)["color"], linestyle="--",
                    label=f"Training Accuracy Maximum = {train_maximum:.4f}")
        plt.axhline(y=val_maximum, c=next(colors)["color"], linestyle="--",
                    label=f"Validation Accuracy Maximum = {val_maximum:.4f}")

        plt.annotate("",
                    xy=(last_key, train_maximum), xycoords='data',
                    xytext=(last_key, val_maximum), textcoords='data',
                    arrowprops=dict(arrowstyle='<|-|>',
                                    shrinkA=0.0,
                                    shrinkB=0.0,
                                    connectionstyle="arc3"),
                    )
        text_offset = matplotlib.transforms.offset_copy(plt.gca().transData, plt.gcf(), x=16, y=-6, units='points')
        plt.text(last_key, val_maximum, f"$\\frac{{val}}{{train}}={val_maximum/train_maximum:.4f}$",
                 va='top', ha='right', transform=text_offset)

        plt.xlabel("Number of Training Steps")
        plt.ylabel("Accuracy")
        plt.title(f"Training and Validation Accuracy ({run_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"task3/task3_accuracies_{name_id}.png")
        plt.show()

    # ================================ PLOT VALIDATION ACCURACY COMPARISON ================================

    included = ["standard", "winit", "isig_solo", "momentum_solo",
                "isig", "winit_momentum", "isig_momentum",
                "momentum"]

    lr_scheduled = ["standard_lrscheduled", "winit_lrscheduled", "isig_lrscheduled", "momentum_lrscheduled"]

    all_history = {k: v for k, v in all_history.items() if k in included}

    _ = plt.figure(figsize=(10, 5))
    for name_id, history in all_history.items():
        run_name = run_names[name_id]
        utils.plot_loss(history['val']["accuracy"], f"+ {run_name}")

    plt.ylim([.9, 1.0025])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy, Run Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"task3/task3_val_accuracies_comparison.png")
    plt.show()

    _ = plt.figure(figsize=(10, 5))
    for name_id, history in all_history.items():
        run_name = run_names[name_id]
        utils.plot_loss(history['val']["accuracy"], f"+ {run_name}")

    plt.ylim([.97, .976])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy, Run Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"task3/task3_val_accuracies_comparison_topzoom.png")
    plt.show()

    # ================================= PLOT TRAINING ACCURACY COMPARISON =================================
    _ = plt.figure(figsize=(10, 5))
    for name_id, history in all_history.items():
        run_name = run_names[name_id]
        utils.plot_loss(history['train']["accuracy"], f"+ {run_name}")

    plt.ylim([.9, 1.0025])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy, Run Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"task3/task3_train_accuracies_comparison.png")
    plt.show()

    # ================================ PLOT VALIDATION LOSS COMPARISON ================================
    _ = plt.figure(figsize=(10, 5))
    for name_id, history in all_history.items():
        run_name = run_names[name_id]
        utils.plot_loss(history['val']["loss"], f"+ {run_name}")

    plt.ylim([0.0, 0.35])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.title(f"Validation Loss, Run Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"task3/task3_val_loss_comparison.png")
    plt.show()

    # ================================= PLOT TRAINING LOSS COMPARISON =================================
    _ = plt.figure(figsize=(10, 5))
    for name_id, history in all_history.items():
        run_name = run_names[name_id]
        utils.plot_loss(history['train']["loss"], f"+ {run_name}", npoints_to_average=25)

    plt.ylim([0.0, 0.35])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.title(f"Training Loss, Run Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"task3/task3_train_loss_comparison.png")
    plt.show()

