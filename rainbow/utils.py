import numpy as np
import datetime
import os
# import plot tools
import matplotlib
import matplotlib.pyplot as plt


color_list = ['b', 'g', 'r', 'y', 'k', 'm']


def preprocess(atari_screen, frames, binary=False):
    """
    Process an 210x160 RGB image into an 80x80 GreyScale Image
    """
    if len(atari_screen) < 4:
        while len(atari_screen) < 4:
            atari_screen.append(atari_screen[-1])
    atari_screen = np.array(atari_screen)
    out = atari_screen[:, 35:195, :]      # crop into square
    out = out[:, ::2, ::2, 0]   # downsample by factor of 2 with one channel
    out[out == 144] = 0     # erase background 1
    out[out == 109] = 0     # erase background 2
    out[out != 0] = 1       # everything else (paddles, ball) just set to 1
    if binary:
        return out.reshape(1, 4, 80, 80).astype(np.bool).tolist()
    else:
        return out.reshape(1, 4, 80, 80)


def plot_var_history(var_history, labels, show_confidence=False,
                     color_list=color_list, x_label='', y_label='',
                     y_ticks=None, log_scale=False, fig_size=(12, 6)):
    """
    Plot the value of a variable at each episode averaged over the number
    of runs at different setting

    Arguments:
        var_history - list/array of the below shape
                      (no. of settings, no. of runs, no.episodes) or
                      (no. of runs, no.episodes)
    """
    if not isinstance(var_history, np.ndarray):
        var_history = np.array(var_history)
    if var_history.ndim == 2:
        var_history = np.expand_dims(var_history, 0)
    assert var_history.ndim == 3 or var_history.ndim == 2, "invalid input"
    # Get mean over all runs
    var_means = np.mean(var_history, axis=1)
    fig, ax = plt.subplots()
    fig.set_size_inches(*fig_size)
    # Graph foramtting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    if log_scale:
        ax.set_yscale("log")
    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Plot values over different setting
    for plot_no in range(var_means.shape[0]):
        ax.plot(var_means[plot_no], color=color_list[plot_no],
                label=labels[plot_no])
        if show_confidence:
            # calculate the 95 percent confidence interval
            num_runs = np.size(var_means, 1)
            episodes = np.arange(1, np.size(var_means, -1), 1)
            reward_ci = 1.960 * (np.std(var_history, axis=1)/np.sqrt(num_runs))
            ax.fill_between(
                            episodes,
                            var_means[plot_no]-reward_ci[plot_no],
                            var_means[plot_no]+reward_ci[plot_no],
                            color=color_list[plot_no], alpha=0.2)
    # Enable legend
    ax.legend()


def get_model_name(param_dict, path):
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    write_path = path + '/' + time_stamp
    os.makedirs(write_path)
    with open(path + '/model_details.txt', 'w+') as info:
        info.write('Model Details:\n')
        for keys, value in param_dict.items():
            info.write(str(keys) + ': ' + str(value) + '\n')
    return write_path
