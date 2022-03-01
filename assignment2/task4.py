import os
import pickle
import time

import matplotlib.transforms as mt
import matplotlib.pyplot as plt
import numpy as np

import utils
from task2 import SoftmaxTrainer
from task2a import one_hot_encode, pre_process_images, SoftmaxModel
from tqdm import tqdm

if __name__ == "__main__":
    num_epochs = 50
    learning_rate = .01
    batch_size = 32
    momentum_gamma = .9
    shuffle_data = True


    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    stochastic_subset_selection = True
    load_percent = 100
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist(600 * load_percent, 100 * load_percent,
                                                           stochastic_subset_selection)

    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)


    def make_trainer(hidden_layers: list[int]):
        return SoftmaxTrainer(
            momentum_gamma,
            use_momentum, None,
            SoftmaxModel([*hidden_layers, 10], use_improved_sigmoid, use_improved_weight_init),
            learning_rate,
            batch_size,
            shuffle_data,
            X_train, Y_train,
            X_val, Y_val,
        )


    # For ease visualisation, we store aside the history of the training,
    # letting us change around plotting parameters later without having to re-train the models
    try_load_history = True
    history_dumpfile = f"task4_{load_percent}pct_{'' if stochastic_subset_selection else 'sequential_'}history.pkl"

    all_history = dict()
    if try_load_history and os.path.exists(history_dumpfile):
        print("Loading history from file...")
        all_history = pickle.load(open(history_dumpfile, 'rb'))

    # This will be thrown away quickly if we find that all the models have already been trained
    # and exist the history. This way we can easily have new trainers added to the list without
    # having to re-train ALL the models
    all_trainers = {
        'baseline': make_trainer([64]),
        '4a': make_trainer([32]),
        '4b': make_trainer([128]),
        '4d': make_trainer([60, 60]),
        '4e': make_trainer([64] * 10),
        'wide': make_trainer([512]),
    }
    # Keep only the trainers we don't have in history
    all_trainers = {k: all_trainers[k] for k in all_trainers if k not in all_history}
    if len(all_trainers) > 0:
        for name, trainer in all_trainers.items():
            print(f"Training {name}...")
            time_start = time.perf_counter()
            train_history, val_history = trainer.train(num_epochs)
            time_elapsed = time.perf_counter() - time_start
            all_history[name] = dict(train=train_history, val=val_history, training_time=time_elapsed)
            print(f"Time elapsed: {time_elapsed:.2f}s")

        pickle.dump(all_history, open(history_dumpfile, "wb"))

    # Plotting
    individual_runs = [
        dict(run='baseline', description='3c: 1 hidden layer, 64 units. 50 880 params',
             metrics=dict(loss=dict(ylim=(0., .5), ), accuracy=dict(ylim=(.9, 1.0025), ))),
        dict(run='4a', description='4a: 1 hidden layer, 32 units. 25 440 params',
             metrics=dict(loss=dict(ylim=(0., .5), ), accuracy=dict(ylim=(.9, 1.0025), ))),
        dict(run='4b', description='4b: 1 hidden layer, 128 units. 101 760 params',
             metrics=dict(loss=dict(ylim=(0., .5), ), accuracy=dict(ylim=(.9, 1.0025), ))),
        dict(run='4d', description='4d: 2 hidden layers, 60 units. 51 300 params',
             metrics=dict(loss=dict(ylim=(0., .5), ), accuracy=dict(ylim=(.9, 1.0025), ))),
        dict(run='4e', description='4e: 10 hidden layers, 64 units. 87 744 params',
             metrics=dict(loss=dict(ylim=(0., .5), ), accuracy=dict(ylim=(.9, 1.0025), ))),
        dict(run='wide', description='wide: 1 hidden layers, 512 units. 407 040 params',
             metrics=dict(loss=dict(ylim=(0., .5), ), accuracy=dict(ylim=(.9, 1.0025), ))),
    ]
    comparison_runs = [
        dict(evaluation='train', runs=('baseline', '4d'), description='4d vs Baseline',
             metrics=dict(loss=dict(ylim=(0., .4), ), accuracy=dict(ylim=(.9, 1.0025), ))),
        dict(evaluation='val', runs=('baseline', '4d'), description='4d vs baseline',
             metrics=dict(loss=dict(ylim=(0., .3), ), accuracy=dict(ylim=(.92, .99), ))),
        dict(evaluation='train', runs=('baseline', '4e'), description='4e vs Baseline',
             metrics=dict(loss=dict(ylim=(0., .4), ), accuracy=dict(ylim=(.9, 1.0025), ))),
        dict(evaluation='val', runs=('baseline', '4e'), description='4e vs baseline',
             metrics=dict(loss=dict(ylim=(0., .3), ), accuracy=dict(ylim=(.92, .99), ))),
        dict(evaluation='val', runs=('baseline', '4a', '4b', '4d', '4e', 'wide'), description='final comparison',
             metrics=dict(loss=dict(ylim=(0., .3), ), accuracy=dict(ylim=(.92, .99), ))),
    ]

    if not os.path.exists('./task4'):
        os.mkdir('task4')

    for setup in individual_runs:
        run_name = setup['run']
        metrics = setup['metrics']
        run_description = setup['description']
        run = all_history[run_name]
        train_history, val_history, training_time = run['train'], run['val'], run['training_time']

        # Plot training and validation loss
        if 'loss' in metrics:
            _ = plt.figure(figsize=(10, 5))
            ylim = metrics['loss']['ylim']
            plt.ylim(ylim)

            utils.plot_loss(train_history['loss'], f"Training Loss ({run_name})", 25)
            utils.plot_loss(val_history['loss'], f"Validation Loss ({run_name})")

            colors = plt.rcParams["axes.prop_cycle"]() # Skip ahead with color cycling
            next(colors)
            next(colors)

            last_key = max(val_history['loss'].keys())
            min_key = min(val_history['loss'], key=val_history['loss'].get)
            final_loss = val_history['loss'][last_key]
            min_loss = val_history['loss'][min_key]

            plt.axhline(y=min_loss, c=next(colors)["color"], linestyle="--",
                        label="Validation Loss Minimum = {:.4f}".format(min_loss))
            plt.axhline(y=final_loss, c=next(colors)["color"], linestyle="--",
                        label="Final Validation Loss = {:.4f}".format(final_loss))
            plt.plot(min_key, min_loss, 'o', c=next(colors)['color'],
                     label=f"Minimum Validation Loss, step {min_key}")
            plt.annotate("",
                         xy=(last_key, final_loss), xycoords='data',
                         xytext=(last_key, min_loss), textcoords='data',
                         arrowprops=dict(arrowstyle='<|-|>',
                                         shrinkA=0.0,
                                         shrinkB=0.0,
                                         connectionstyle="arc3"),
                         )
            text_offset = mt.offset_copy(plt.gca().transData, plt.gcf(), x=16, y=-6, units='points')
            plt.text(last_key, min_loss, f"$final - minimum={final_loss - min_loss:.4f}$",
                     va='top', ha='right', transform=text_offset)

            secax = plt.gca().secondary_xaxis('top', functions=(lambda x: training_time * (x / last_key),
                                                                lambda x: x / training_time))
            secax.set_xlabel('Training Time (s)', labelpad=-12)
            secax.set_xticks([0, training_time])
            secax.set_xticklabels(['0s', f'{training_time:.2f}s'])

            plt.xlabel("Number of Training Steps")
            plt.ylabel("Cross Entropy Loss - Average")
            plt.title(f"Losses [{run_description}] evaluating on {load_percent}% of datasets, "
                      f"{'stochastic' if stochastic_subset_selection else 'sequential'} subset")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"task4/task4_losses_{load_percent}pct_"
                        f"{'stochastic' if stochastic_subset_selection else 'sequential'}_{run_name}.png")
            plt.show()

        if 'accuracy' in metrics:
            _ = plt.figure(figsize=(10, 5))
            ylim = metrics['accuracy']['ylim']
            plt.ylim(ylim)

            utils.plot_loss(train_history['accuracy'], f"Training Accuracy ({run_name})")
            utils.plot_loss(val_history['accuracy'], f"Validation Accuracy ({run_name})")

            colors = plt.rcParams["axes.prop_cycle"]()  # Skip ahead with color cycling
            next(colors)
            next(colors)

            last_key = max(val_history['loss'].keys())
            val_maximum = max(val_history['accuracy'].values())
            train_maximum = max(train_history['accuracy'].values())

            plt.axhline(y=val_maximum, c=next(colors)["color"], linestyle="--",
                        label="Validation Accuracy Maximum = {:.4f}".format(val_maximum))
            plt.axhline(y=train_maximum, c=next(colors)["color"], linestyle="--",
                        label="Training Accuracy Maximum = {:.4f}".format(train_maximum))

            plt.annotate("",
                         xy=(last_key, train_maximum), xycoords='data',
                         xytext=(last_key, val_maximum), textcoords='data',
                         arrowprops=dict(arrowstyle='<|-|>',
                                         shrinkA=0.0,
                                         shrinkB=0.0,
                                         connectionstyle="arc3"),
                         )
            text_offset = mt.offset_copy(plt.gca().transData, plt.gcf(), x=16, y=-6, units='points')
            plt.text(last_key, val_maximum, f"$\\frac{{val}}{{train}}={val_maximum / train_maximum:.4f}$",
                     va='top', ha='right', transform=text_offset)

            secax = plt.gca().secondary_xaxis('top', functions=(lambda x: training_time * (x / last_key),
                                                                lambda x: x / training_time))
            secax.set_xlabel('Training Time (s)', labelpad=-12)
            secax.set_xticks([0, training_time])
            secax.set_xticklabels(['0s', f'{training_time:.2f}s'])

            plt.xlabel("Number of Training Steps")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracies [{run_description}] evaluating on {load_percent}% of datasets, "
                      f"{'stochastic' if stochastic_subset_selection else 'sequential'} subset")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"task4/task4_accuracies_{load_percent}pct_"
                        f"{'stochastic' if stochastic_subset_selection else 'sequential'}_{run_name}.png")
            plt.show()

    for setup in comparison_runs:
        metrics = setup['metrics']
        comparison_description = setup['description']
        evaluation = setup['evaluation']
        runs = setup['runs']

        active_form = {'train': 'Training', 'val': 'Validation'}
        if 'loss' in metrics:
            _ = plt.figure(figsize=(10, 5))
            plt.ylim(metrics['loss']['ylim'])

            for run_name in runs:
                run = all_history[run_name][evaluation]
                training_time = all_history[run_name]['training_time']
                utils.plot_loss(run['loss'], f"{active_form[evaluation]} Loss ({run_name})", 25 if evaluation == 'train' else 1)

            # max_key = max(run['accuracy'].keys())
            # secax = plt.gca().secondary_xaxis('top', functions=(lambda x: training_time * (x / max_key),
            #                                                     lambda x: x / training_time))  # type: plt.Axes
            # secax.set_xlabel('Training Time (s)', labelpad=-12)
            # secax.set_xticks([0, training_time])
            # secax.set_xticklabels(['0s', f'{training_time:.2f}s'])

            plt.xlabel(f"Number of Training Steps")
            plt.ylabel("Cross Entropy Loss - Average")
            plt.title(f"Losses [{comparison_description}] evaluating on {load_percent}% of datasets, "
                      f"{'stochastic' if stochastic_subset_selection else 'sequential'} subset")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"task4/task4_{evaluation}_losses_{load_percent}pct_"
                        f"{'stochastic' if stochastic_subset_selection else 'sequential'}_{'_vs_'.join(runs)}.png")
            plt.show()

        if 'accuracy' in metrics:
            _ = plt.figure(figsize=(10, 5))
            ylim = metrics['accuracy']['ylim']
            plt.ylim(ylim)

            colors = plt.rcParams["axes.prop_cycle"]()
            for run_name in runs:
                run = all_history[run_name][evaluation]
                training_time = all_history[run_name]['training_time']
                utils.plot_loss(run['accuracy'], f"{active_form[evaluation]} Accuracy ({run_name})")
                max_accuracy = max(run['accuracy'].values())

                plt.axhline(y=max_accuracy, c=next(colors)["color"], linestyle="--",
                            label=f"{active_form[evaluation]} Accuracy Maximum = {max_accuracy:.4f}")
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracies [{comparison_description}] evaluating on {load_percent}% of datasets, "
                      f"{'stochastic' if stochastic_subset_selection else 'sequential'} subset")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"task4/task4_{evaluation}_accuracies_{load_percent}pct_"
                        f"{'stochastic' if stochastic_subset_selection else 'sequential'}_{'_vs_'.join(runs)}.png")
            plt.show()

            # Plot wall-clock time accuracy progression
            _ = plt.figure(figsize=(10, 5))
            ylim = metrics['accuracy']['ylim']
            plt.ylim(ylim)

            colors = plt.rcParams["axes.prop_cycle"]()
            for run_name in runs:
                run = all_history[run_name][evaluation]
                training_time = all_history[run_name]['training_time']
                last_key = max(run['accuracy'].keys())
                max_accuracy = max(run['accuracy'].values())
                ys = np.array(list(run['accuracy'].values()))
                xs = np.array(list(run['accuracy'].keys()))
                xs = (xs / last_key) * training_time

                plt.plot(xs, ys, label=f"{active_form[evaluation]} Accuracy ({run_name})")

                plt.axhline(y=max_accuracy, c=next(colors)["color"], linestyle="--",
                            label=f"{active_form[evaluation]} Accuracy Maximum = {max_accuracy:.4f}")

            plt.xlabel("Time (s)")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracies Time-wise [{comparison_description}] evaluating on {load_percent}% of datasets, "
                      f"{'stochastic' if stochastic_subset_selection else 'sequential'} subset")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"task4/task4_{evaluation}_accuracies_{load_percent}pct_"
                        f"{'stochastic' if stochastic_subset_selection else 'sequential'}_{'_vs_'.join(runs)}_time.png")
            plt.show()


