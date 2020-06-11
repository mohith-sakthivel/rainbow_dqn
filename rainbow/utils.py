import numpy as np
# import plot tools
import matplotlib
import matplotlib.pyplot as plt


color_list = ['b', 'g', 'r', 'y', 'k', 'm']


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


def get_model_name(param_dict):
    hid_lyrs = ''.join(str(param_dict.get("hid_lyrs")).split(','))
    lr = param_dict.get('lr', 'x')
    target_freq = param_dict.get('target_update_freq', 'x')
    mini_batch = param_dict.get('mini_batch', 'x')

    name = 'hid_{0}_lr_{1}_target_freq_{2}_mini_batch_{3:.0e}'.\
        format(hid_lyrs, lr, target_freq, mini_batch)
    return name
