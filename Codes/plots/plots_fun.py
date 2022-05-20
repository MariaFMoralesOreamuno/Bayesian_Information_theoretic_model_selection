from log import *

"""
Author: María Fernanda Morales Oreamuno 

Created: 31/05/2021
Last update: 12/12/2021

Module contains the main plot properties (used by all plots) as well as the basic functions used for plot formatting. 
Module also contains all functions to plot and save the different results for the analytical models and all BMJ results
"""
# Set font sizes:
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 11

# Set bar width:
width = 0.75
# model_colors = ['royalblue', 'gold', 'mediumseagreen', 'coral', 'olive', 'cadetblue']
param_colors = ['royalblue', 'gold', 'mediumseagreen', 'coral', 'olive', 'red', 'yellow', 'black', 'green', 'pink']

model_colors = ['#2c7bb6', '#fee090', '#fc8d59', '#d73027', '#91bfdb', '#e0f3f8']

# LaTex document properties
textwidth = 448
textheight = 635.5
plt.rc("text", usetex='True')
plt.rc('font', family='serif', serif='Arial')

# Set plot properties:
# plt.rc('figure', figsize=(15, 15))
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title
plt.rc('axes', axisbelow=True)

plt.rc('savefig', dpi=1000)

# global variables
score_list = ["-log(BME)", "NNCE", "RE", "IE"]
score_list_2 = ["log(BME)", "ELPD", "RE", "IE"]
subplot_titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]
markers = ["o", "s", "X", '^', '*']


# Basic Format functions ---------------------------------------------------------------------------------
def plot_size(fraction=1.0, rows=2):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
        Parameters
        ----------
        :param fraction: float, optional (Fraction of the width which you wish the figure to occupy)
        :param rows = int, optional (Number of rows in the plot)

        Returns
        -------
        fig_dim: tuple (Dimensions of figure in inches)
        """
    # Width of figure (in pts)
    fig_width_pt = textwidth * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # if height_fraction == 0:
    #     # Figure height in inches: golden ratio
    #     fig_height_in = fig_width_in * golden_ratio
    # else:
    #     fig_height_in = textheight * inches_per_pt * height_fraction

    if rows <= 2:
        # Figure height in inches: golden ratio
        fig_height_in = fig_width_in * golden_ratio
    else:
        if rows == 3:
            h_fraction = 0.65
        elif rows == 4:
            h_fraction = 0.85
        else:
            h_fraction = 1
        fig_height_in = textheight * inches_per_pt * h_fraction

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def get_limits(data_list):
    """
    Given a list of numpy arrays, function finds the max and min value among all arrays

    Args:
    ---------------------------
    :param data_list: list with np.arrays

    :return: floats with max and min value
    """
    min_v = 0
    max_v = 0
    for elem in data_list:
        if np.nanmax(elem) > max_v:
            max_v = np.nanmax(elem)
        if np.nanmin(elem) < min_v:
            min_v = np.nanmin(elem)
    return max_v, min_v


def get_multiple_bar_location(w, num_bars):
    """
    Function calculates the location each bar in a multiple bar graph.

    Args:
    -----------------------------
    :param w: float, width of bars
    :param num_bars: int, number of bars per x value
    ----------------------------------
    :return: np.array with location of center of each bar
    """
    if num_bars == 3:
        loc = np.arange(-w / num_bars, (w / num_bars) * 2, w / num_bars)
    elif num_bars == 2:
        loc = np.arange(-w / (num_bars * 2), w / (num_bars * 2) * 2, w / num_bars)
    return loc


# General plotting functions
def plot_bar(axis, x, y, index, uncertainty, value_label=False):
    """
    Function plots bar graphs.

    Args:
    :param axis: plot axis
    :param x: list, x value labels (e.g. model names)
    :param y: np.array, y values (score values for each model)
    :param index: int, to get y axis label
    :param uncertainty: boolean, when True uses score names with entropy, when False uses predictive capability related
    labels
    :param value_label: boolean, when True, adds label to each bar

    Note: User has the option to add value labels over each bar.
    """
    # Generate bar plot:
    x_loc = np.arange(1, len(x) + 1)

    bar_plot = axis.bar(x_loc, y, width=width, color=model_colors[index])

    axis.set_xlabel("Model")
    if uncertainty:
        axis.set_ylabel(score_list[index])
    else:
        axis.set_ylabel(score_list_2[index])

    # Add y axis label to each bar
    def add_label(bar):
        """
        Add labels to each bar in the graph
        :param bar: bar plot
        :return:
        """
        for i, rect in enumerate(bar):
            height = rect.get_height()
            if abs(height) < 0.001:
                label = str("{:.2e}".format(height))
            else:
                label = str(round(height, 2))
            # print("Height: ", height, " in scientific notation: ", label)
            axis.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, label, ha="center", va="bottom", rotation=0,
                      size=SMALL_SIZE - 1)

    # Y limits (to focus on the differences)
    if (np.max(y) - np.min(y)) > 50:
        factor = 5
    elif np.max(y) - np.min(y) < 1:
        factor = 0.25
    else:
        factor = 0.5
    plt.ylim([np.min(y) - factor, np.max(y) + factor])
    # if np.all(y < 0):  # If negative values
    #     # plt.ylim([np.max(y) + 0.5, np.min(y) - 0.5])
    #     plt.ylim([np.max(y) + factor, np.min(y) - factor])
    # else:
    #     # plt.ylim([np.min(y) - 0.5, np.max(y) + 0.5])
    #     plt.ylim([np.min(y) - factor, np.max(y) + factor])

    # x-Ticks: set one tick for each bar
    axis.set_xticks(np.arange(1, len(x) + 1))
    axis.set_xticklabels(x)

    if value_label:
        add_label(bar_plot)

    return axis


# BMS ----------------------------------------------------------------------------------------
def plot_scores_bar(score_array, model_names, save_name, uncertainty):
    """
    Function plots each BMS score for each model as bar graphs. Each BMS score is graphed in a different subplot, for a
    total of 4 subplots in the main figure

    Args:
    ----------
    :param score_array: array with scores [MxS] where S=number of scores (4) and M = number of models. The score order
    is: BME, NNCE, RE, IE
    :param model_names: Array with shape [1xM] with model names, as strings
    :param save_name: file path + name.ext with which to save the resulting figure
    :param uncertainty: boolean, if true it plots -log(BME) and NNCE, if false it plots BME and ELPD
    ----------
    :return: Save plots (no return)

    Note: function calls another function to generate the bar graphs in each loop.
    """

    # plt.figure(1, constrained_layout=True)
    plt.figure(1, figsize=plot_size())

    # Loop through each score
    for i in range(0, score_array.shape[1]):
        p = plt.subplot(2, 2, i + 1)

        data = score_array[:, i]
        if not uncertainty and (i == 0 or i == 1):
            data = -1 * data

        p = plot_bar(p, model_names, data, i, uncertainty)
        if i < score_array.shape[1] / 2:
            p.set_xlabel("")
        p.set_title(subplot_titles[i], loc='left', fontweight='bold')
        p.grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
        # p.text(1, 1, subplot_titles[i])

    # Format:
    plt.subplots_adjust(top=0.92, bottom=0.1, wspace=0.45, hspace=0.3)
    plt.margins(y=0.5, tight=True)

    # Save plot:
    plt.savefig(save_name)

    # Print
    plt.show(block=False)
    x = 1


def plot_individual_scores(score_array, model_names, file_path, uncertainty):
    """
    Function plots each BMS score for each model as bar graphs. Each BMS score is graphed and saved in a different plot

    Args:
    ----------
    :param score_array: array with scores [MxS] where M = number of models and S=number of scores (up to 4). The score
    order should be: BME, ELPD, RE, IE
    :param model_names: Array with shape [1xM] with model names, as strings
    :param file_path: file path (folder and prefix)) with which to save the resulting figure
    :param uncertainty: boolean, if true it plots -log(BME) and NNCE, if false it plots BME and ELPD
    ----------
    :return: Save plots (no return)

    Note: function calls another function to generate the bar graphs in each loop.
    """

    for i in range(0, score_array.shape[1]):
        fig, p = plt.subplots(1, figsize=plot_size(fraction=0.75))  # Each plot will be half a LaTex page

        data = score_array[:, i]
        if not uncertainty and (i == 0 or i == 1):
            data = -1 * data

        p = plot_bar(p, model_names, data, i, uncertainty)

        plt.tight_layout()

        if uncertainty:
            save_name = os.path.join(f'{file_path}_{score_list[i]}.eps')
        else:
            save_name = os.path.join(f'{file_path}_{score_list_2[i]}.eps')
        plt.savefig(save_name)

        plt.show(block=False)
        x = 1


def plot_bme(bme_array, model_names, save_name):
    if bme_array.ndim == 2:
        bme_array = bme_array[:, 0]

    # fig, ax = plt.subplots(figsize=plot_size(fraction=0.5))
    fig, ax = plt.subplots(figsize=[3.1, 3.5])

    x_loc = np.arange(1, len(model_names) + 1)

    bar_plot = ax.bar(x_loc, bme_array, width=width, color=model_colors[0])

    ax.set_xlabel("Model")
    ax.set_ylabel("BME", labelpad=-10)

    plt.yscale("log")
    ax.grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')

    # Add y axis label to each bar
    def add_label(bar):
        """
        Add labels to each bar in the graph
        :param bar: bar plot
        :return:
        """
        for i, rect in enumerate(bar):
            height = rect.get_height()
            if abs(height) < 0.001:
                label = str("{:.2e}".format(height))
            else:
                label = str(round(height, 2))
            # print("Height: ", height, " in scientific notation: ", label)
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, label, ha="center", va="bottom", rotation=0,
                    size=SMALL_SIZE - 1)

    # Y limits (to focus on the differences)
    y_min = np.min(bme_array)
    y_max = np.max(bme_array)

    plt.ylim([y_min * 0.5, y_max * 1.5])

    # x-Ticks: set one tick for each bar
    ax.set_xticks(np.arange(1, len(model_names) + 1))
    ax.set_xticklabels(model_names)

    add_label(bar_plot)

    # Save plot:
    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


def plot_multiple_score_weights(data, x, stack_labels, path):
    """
    Function plots the model weights for multiple scores, as stacked-bar plots. Each score will be plot in a different
    subplot

    Args:
    ----------------
    :param data: list with S np.arrays, one for each score to plot. Each np.array are of sizeNxM size, where N is the
    number of x values (number of data points) and M is the numberof models (number of stack_label values)
    :param x: list with x axis values (number of data points analyzed, number of rows in the input data array)
    :param stack_labels: list with name of models being analyzed (length=number of columns in input data array)
    :param path: file path to save the resulting plot
    """
    fig, ax = plt.subplots(figsize=plot_size())
    for k, score in enumerate(data):  # Loop through each score
        ax = plt.subplot(2, 2, k + 1)
        for i in range(0, score.shape[1]):  # Loop through each model
            if i == 0:
                b = np.full(x.shape[0], 0)
            else:
                b = b + score[:, i - 1]
            ax.bar(x.astype(str), score[:, i], width=width, label=stack_labels[i], bottom=b, color=model_colors[i])
        # Y axis:
        plt.ylim([0, 1.1])
        ax.set_ylabel(score_list[k] + "model weight")
        # X axis
        ax.set_xlabel("Number of data points")

        if k + 1 <= len(data) / 2:
            ax.set_xlabel("")  # Remove x label from the upper graphs, for legibility
        handles, labels = ax.get_legend_handles_labels()
        ax.set_title(subplot_titles[k], loc='left', fontweight='bold')

    # Format:
    plt.subplots_adjust(top=0.95, bottom=0.1, wspace=0.35, hspace=0.25)
    plt.margins(y=1, tight=True)

    fig.legend(handles, labels, loc='upper right', title="Model")
    plt.show(block=False)


def plot_scores_calculations(data, ce, labels, save_name):
    """
    Function plots the scores for all models in 2 bar plots. The first subplot contains the log(BME), NNCE  and RE, to
    show how you ge RE from elpd and log(BME). The second subplot contains the RE, cross entropy (expected posterior
    density) and IE.

    Args:
    --------------------------
    :param data: np.array with all score values, with shape [MxS]
    :param ce: np.array with cross entropy, or expected posterior density values for each model
    :param labels: np.array with model names, to serve as labels in the x-axis
    :param save_name: path, as string, of file where to save the resulting plot
    """

    # Make sure all arrays are 2D
    if data.ndim == 1:
        data = np.reshape(data, (1, data.shape[0]))
        ce = np.reshape(np.array([ce]), (1, 1))
        labels = np.array([labels])

    # Abs. value of all values   - Uncomment for absolute value
    # data[:, 1] = np.abs(data[:, 1])
    # post_density = np.abs(post_density)
    # Add density

    if np.max(np.abs(data)) > np.min(np.abs(data)) + 50:
        sy = False
    else:
        sy = True

    # Plot (BME, elpd, RE) in one, and (RE, density, IE) in the other
    fig, ax = plt.subplots(2, 1, figsize=plot_size(), sharey=sy)
    num_bars = 3

    # Get location of each plot (for each model)
    x = np.arange(len(labels))
    loc = get_multiple_bar_location(width, num_bars)

    # Top plot
    for s in range(0, 3):
        ax[0].bar(x + loc[s], data[:, s], label=score_list[s], color=model_colors[s], width=width / num_bars)

    ax[0].set_ylabel('Scores')
    ax[0].set_title(subplot_titles[0] + ": RE = -log(BME) - NNCE", loc='left', fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    handles, lab = ax[0].get_legend_handles_labels()

    # Bottom plot:
    ax[1].bar(x + loc[0], data[:, 2], label=score_list[2], color=model_colors[2], width=width / num_bars)
    ax[1].bar(x + loc[1], -ce[:, 0], label='CE', color=model_colors[4], width=width / num_bars)
    ax[1].bar(x + loc[2], data[:, 3], label=score_list[3], color=model_colors[3], width=width / num_bars)

    ax[1].set_ylabel('Scores')
    ax[1].set_title(subplot_titles[1] + ": IE = CE - RE", loc='left', fontweight='bold')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_xlabel('Model')
    handles_1, lab_1 = ax[1].get_legend_handles_labels()

    handles = handles + handles_1
    del handles[2]
    labls = list(dict.fromkeys(lab + lab_1))

    fig.legend(handles, labls, loc='lower center', ncol=5)

    ax[0].grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
    ax[1].grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
    plt.subplots_adjust(top=0.92, bottom=0.2, wspace=0.1, hspace=0.4)
    plt.margins(y=0.5, tight=True)

    # Save plot:
    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


def plot_stacked_score_calculation(data, ce, labels, save_name):
    """
        Function plots the scores for all models in 2 bar plots. The first subplot contains the -log(BME) and -elpd in
        a stacked bar plot to determine how to obtain RE from these values. The second subplot contains the RE and cross
        entropy as a stacked bar plot to determine how to obtain IE.

        Args:
        --------------------------
        :param data: np.array with all score values, with shape [MxS]
        :param ce: np.array with cross entropy, or expected posterior density values for each model
        :param labels: np.array with model names, to serve as labels in the x-axis
        :param save_name: path, as string, of file where to save the resulting plot
        """

    # Make sure all arrays are 2D
    if data.ndim == 1:
        data = np.reshape(data, (1, data.shape[0]))
        ce = np.reshape(np.array([ce]), (1, 1))
        labels = np.array([labels])

    # Abs. value of all values   - Uncomment for absolute value
    # data[:, 1] = np.abs(data[:, 1])
    # post_density = np.abs(post_density)
    # Add density

    if np.max(np.abs(data)) > np.min(np.abs(data)) + 50:
        sy = False
    else:
        sy = False

    # Plot (BME, elpd, RE) in one, and (RE, density, IE) in the other
    fig, ax = plt.subplots(2, 1, figsize=plot_size(), sharey=sy)
    num_bars = 2

    # Get location of each plot (for each model)
    x = np.arange(len(labels))
    loc = get_multiple_bar_location(width, num_bars)

    # Function to get line width
    def get_plot_range(max_v, min_v):
        """
        Function determines the plot y range, depending on the value and sign of the data ranges.

        Args:
        :param max_v: float, with maximum number
        :param min_v: float, with minimum number

        :return: floats, modified max and min values
        """
        if max_v >= 100 or min_v <= -100:
            max_v = max_v + 25
            min_v = min_v - 25
        elif 100 < max_v < 50 or -100 < min_v < -50:
            max_v = max_v + 10
            min_v = min_v - 10
        elif 50 >= max_v > 1 or -50 <= min_v <= -1:
            max_v = max_v + 2
            min_v = min_v - 2
        else:
            max_v = max_v + 0.5
            min_v = min_v - 0.5
        return max_v, min_v

    # Top plot ------------------------------------------------------------------------------------------ #
    # Line thickness:
    el_lw = np.where(data[:, 1] < 0, 3, 1)
    # get limits
    max_0, min_0 = get_limits([data[:, 0], data[:, 1], data[:, 2]])
    max_0, min_0 = get_plot_range(max_0, min_0)

    #   Stacked plot:
    ax[0].bar(x + loc[0], data[:, 0], label=score_list[0], color=model_colors[0], width=width / num_bars,
              linewidth=1, edgecolor=model_colors[0])  # BME
    ax[0].bar(x + loc[1], data[:, 1], label='-ELPD', color=model_colors[1], width=width / num_bars,
              linewidth=el_lw, edgecolor=model_colors[1])  # -elpd
    ax[0].bar(x + loc[1], data[:, 2], label=score_list[2], color=model_colors[2], width=width / num_bars,
              edgecolor=model_colors[2], bottom=data[:, 1])  # RE

    ax[0].set_ylabel('Scores')
    ax[0].set_title(subplot_titles[0] + ": RE = -log(BME) - ELPD", loc='left', fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    # ax[0].set_
    handles, lab = ax[0].get_legend_handles_labels()
    ax[0].set_ylim(min_0, max_0)

    # Bottom plot: ------------------------------------------------------------------------------------------ #
    # Line thickness:
    re_lw = np.where(data[:, 3] < 0, 2, 1)

    # get limits:
    max_1, min_1 = get_limits([data[:, 2], -ce[:, 0], data[:, 3]])
    max_1, min_1 = get_plot_range(max_1, min_1)

    ax[1].bar(x + loc[1], data[:, 2], label=score_list[2], color=model_colors[2], width=width / num_bars,
              bottom=data[:, 3], linewidth=re_lw, edgecolor=model_colors[2])
    ax[1].bar(x + loc[0], -ce[:, 0], label='CE', color=model_colors[4], width=width / num_bars,
              edgecolor=model_colors[4])
    ax[1].bar(x + loc[1], data[:, 3], label=score_list[3], color=model_colors[3], width=width / num_bars,
              edgecolor=model_colors[3])

    ax[1].set_ylabel('Scores')
    ax[1].set_title(subplot_titles[1] + ": IE = CE - RE", loc='left', fontweight='bold')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_xlabel('Model')
    handles_1, lab_1 = ax[1].get_legend_handles_labels()
    ax[1].set_ylim(min_1, max_1)

    handles = handles + handles_1
    del handles[3]
    labls = list(dict.fromkeys(lab + lab_1))

    fig.legend(handles, labls, loc='lower center', ncol=5)

    ax[0].grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
    ax[1].grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
    plt.subplots_adjust(top=0.92, bottom=0.2, wspace=0.1, hspace=0.4)
    plt.margins(y=0.5, tight=True)

    # Save plot:
    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


# BMS scores for different data points: ---------------------------------------------------------------
def plot_bme_weights(data, x, stack_labels, save_name):
    """
    Function plots the BME model weights, as stacked-bar plots.

    Args:
    ----------------
    :param data: np.array with MxN size, where N is the number of x values (number of data points) and M is the number
                 of models (number of stack_label values)
    :param x: list with x axis values (number of data points analyzed, number of rows in the input data array)
    :param stack_labels: list with name of models being analyzed (length=number of columns in input data array)
    :param save_name: file name and path where to save the resulting plot
    """

    fig, ax = plt.subplots(figsize=plot_size(fraction=1))
    for i in range(0, data.shape[0]):
        if i == 0:
            b = np.full(x.shape[0], 0)
        else:
            b = b + data[i - 1, :]
        ax.bar(x.astype(str), data[i, :], width=width, label=stack_labels[i], bottom=b, color=model_colors[i])
    # Y axis:
    plt.ylim([0, 1.1])
    ax.set_ylabel("BME model weight")
    # X axis
    ax.set_xlabel("Number of data points")
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', title="Model", ncol=stack_labels.shape[0])
    fig.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(top=0.95, bottom=0.25, wspace=0.1, hspace=0.4)

    # Save plot:
    plt.savefig(save_name)
    # Plot
    plt.show(block=False)
    x = 1


def plot_evolving_bme(bme_array, bme_weights, num_dp, model_names, save_name):
    """
    Plots how the BME and BME weights change with increasing number of data points. The left plot will contain BME
    values and the right plot BME weights

    Args:
    -------------------------------------
    :param bme_array: np.array of size MxP, where M is number of models, and P is the number of data points being
    analyzed.
    :param bme_weights: np.array of size MxP, with BME weights. The rows for each column add to 1
    :param num_dp: list with the 'x' values (number of data points for which the BI procedure was done)
    :param model_names: list with model names (each curve in each subplot)
    :param save_name: path of file where to save the resulting plot
    """
    # Set figure size:
    cols = 2
    fig, ax = plt.subplots(1, cols, figsize=plot_size())

    x = np.arange(1, len(num_dp) + 1)

    for m in range(0, bme_array.shape[0]):
        y = bme_array[m, :]
        ax[0].plot(num_dp, y, color=model_colors[m], marker=markers[m], linewidth=1, label=model_names[m], markersize=6)

        if m == 0:
            b = np.full(len(num_dp), 0)
        else:
            b = b + bme_weights[m - 1, :]
        ax[1].bar(x, bme_weights[m, :], width=width, label=model_names[m], bottom=b,
                  color=model_colors[m])
    # X Axis
    ax[0].set_xlabel('Number of data points')
    ax[1].set_xlabel('Number of data points')

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(num_dp)

    # y label
    ax[0].set_ylabel("BME value")
    ax[1].set_ylabel("BME weights")
    if np.any(bme_array < 0.001) and np.all(bme_array > 0):
        ax[0].set_yscale("log")
        ax[0].grid(b=True, which='minor', color='gainsboro', linestyle='-')
        # ax[0].set_minorticks_on()
        ax[0].set_ylim(np.min(bme_array) * 0.1, np.max(bme_array) * 10)
    ax[0].grid(b=True, which='major', color='lightgrey', linestyle='-')
    handles, labels = ax[0].get_legend_handles_labels()

    ax[1].set_ylim([0, 1.1])

    ax[0].set_title(subplot_titles[0], loc='left', fontweight='bold')
    ax[1].set_title(subplot_titles[1], loc='left', fontweight='bold')

    # Format:
    plt.subplots_adjust(top=0.95, bottom=0.2, wspace=0.35, hspace=0.25)
    plt.margins(y=1, tight=True)
    # Figure legend
    fig.legend(handles, labels, loc='lower center', ncol=bme_array.shape[0])

    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


def plot_evolving_values(data_array, num_dp, model_names, save_name, uncertainty):
    """
    Plots how the value of each score changes with increasing number of data points. The values for each score are
    plotted in a different subplot. Each subplot contains curves for each model being analyzed. Each score is plotted
    in a different subplot

    Args:
    -------------------------------------
    :param data_array: 3D np.array 'P' 2D arrays, one for each number of data points analyzed. Each 2D array has size
    [MxS], where M is number of models, and S is the different scores.
    :param num_dp: list with the 'x' values (number of data points for which the BI procedure was done)
    :param model_names: list with model names (each curve in each subplot)
    :param save_name: path of file where to save the resulting plot
    :param uncertainty: boolean, if True plots -log(BME) and NNCE, if False plots log(BME) and ELPD

    Note:
    *Plots BME as -log(BME) to have everything in the same scale
    """
    rows = 2
    cols = 2
    # Set figure size:
    fig, ax = plt.subplots(figsize=plot_size())

    for s in range(0, data_array.shape[2]):  # Loop through each score
        data = np.transpose(data_array[:, :, s])
        if not uncertainty and (s == 0 or s == 1):
            data = -1 * data

        ax = plt.subplot(rows, cols, s + 1)

        for m in range(0, data.shape[0]):
            y = data[m, :]
            ax.plot(num_dp, y, color=model_colors[m], marker=markers[m], linewidth=1, label=model_names[m],
                    markersize=6)
        # X Axis
        if s + 1 <= 2:
            ax.set_xlabel("")  # Remove x label from the upper graphs, for legibility
        else:
            ax.set_xlabel("Number of data points")
        # Y axis:
        if uncertainty:
            ax.set_ylabel(score_list[s])
        else:
            ax.set_ylabel(score_list_2[s])

        if s == 0:
            if np.any(data < 0.001) and np.all(data > 0):
                plt.yscale("log")
                plt.grid(b=True, which='minor', color='gainsboro', linestyle='-')
                plt.minorticks_on()
        plt.grid(b=True, which='major', color='lightgrey', linestyle='-')

        # Legend:
        handles, labels = ax.get_legend_handles_labels()

    # Format:
    plt.subplots_adjust(top=0.95, bottom=0.2, wspace=0.35, hspace=0.25)
    plt.margins(y=1, tight=True)
    # Figure legend
    fig.legend(handles, labels, loc='lower center', ncol=data_array.shape[1])

    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


def plot_evolving_calculations(data_array, ce, num_dp, save_name, model_name, all_models=True):
    """
       Plot how the value of each score changes with increasing number of data points. The values for each score are
       plotted in a different subplot. Each subplot contains curves for each model being analyzed. Each score is plotted
       in a different subplot

       Args:
       -------------------------------------
       :param data_array: 3D np.array 'P' 2D arrays, one for each number of data points analyzed. Each 2D array has size
       [MxS], where M is number of models, and S is the different scores.
       :param num_dp: list with the 'x' values (number of data points for which the BI procedure was done)
       :param ce: np.array with cross entropy, or post probability density values for each model, for each 'P'
       data point.
       :param save_name: path of file where to save the resulting plot
       :param all_models: boolean, when True plots for all models, when False plots only for the first Model

       Note:
       *Plots BME as -log(BME) to have everything in the same scale
       """
    cols = 2
    if all_models:
        rows = data_array.shape[1]  # For each model
        fig, ax = plt.subplots(rows, cols, figsize=plot_size(rows=rows), sharey=True)
    else:
        rows = 1
        fig, ax = plt.subplots(rows, cols, figsize=plot_size(), sharey=True)
        ax = np.reshape(ax, (rows, cols))

    # Bottom spacing
    if rows < 3:
        bottom_s = 0.2
    elif 3 <= rows <= 4:
        bottom_s = 0.15
    else:
        bottom_s = 0.1

    i = 0
    for m in range(0, rows):
        # Convert data to 2D (rows: score for different Np, columns: each score)
        data = np.copy(data_array)[:, m, :]
        data[:, 1] = -1 * data[:, 1]
        for s in range(0, data.shape[1]):
            if s <= 2:
                if s != 1:
                    ax[m, 0].plot(num_dp, data[:, s], color=model_colors[s], marker=markers[s], linewidth=1,
                                  label=score_list[s], markersize=6)
                else:
                    ax[m, 0].plot(num_dp, data[:, s], color=model_colors[s], marker=markers[s], linewidth=1,
                                  label='ELPD', markersize=6)
            if s >= 2:
                ax[m, 1].plot(num_dp, data[:, s], color=model_colors[s], marker=markers[s], linewidth=1,
                              label=score_list_2[s], markersize=6)
        # Plot cross entropy
        ax[m, 1].plot(num_dp, np.abs(ce[m, :]), color=model_colors[4], marker=markers[4], linewidth=1,
                      label="CE", markersize=6)

        handles, labels = ax[m, 0].get_legend_handles_labels()
        handles_1, labels_1 = ax[m, 1].get_legend_handles_labels()

        # Properties
        ax[m, 0].set_ylabel('Score values')
        ax[m, 0].grid(b=True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
        ax[m, 1].grid(b=True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)

        if rows > 1:
            ax[m, 0].set_title(subplot_titles[i] + " " + model_name[m], loc='left', fontweight='bold')
            ax[m, 1].set_title(subplot_titles[i + 1] + " " + model_name[m], loc='left', fontweight='bold')
        else:
            ax[m, 0].set_title(subplot_titles[i], loc='left', fontweight='bold')
            ax[m, 1].set_title(subplot_titles[i + 1], loc='left', fontweight='bold')
        i += 2

    ax[m, 0].set_xlabel("Number of data points")
    ax[m, 1].set_xlabel("Number of data points")

    # Legend:
    handles = handles + handles_1
    del handles[2]
    labels = list(dict.fromkeys(labels + labels_1))
    fig.legend(handles, labels, loc='lower center', ncol=5)

    plt.subplots_adjust(top=0.95, bottom=bottom_s, wspace=0.2, hspace=0.4)
    plt.margins(y=0.5, tight=True)

    plt.savefig(save_name)
    plt.show(block=False)
    x = 1
    # ------------------------------------------------------------------------------------------------------------- #
    # cols = 1
    # if all_models:
    #     fig, ax = plt.subplots(rows, cols, figsize=plot_size(height_fraction=1), sharey=True)
    # else:
    #     fig, ax = plt.subplots(rows, cols, figsize=plot_size(), sharey=True)
    #     ax = np.reshape(ax, (rows,))
    #
    # for m in range(0, rows):
    #     # Convert data to 2D (rows: score for different Np, columns: each score)
    #     data = data_array[:, m, :]
    #
    #     for s in range(0, data.shape[1]):
    #         if s < 2:
    #             lw = 1
    #         else:
    #             lw = 2
    #         ax[m].plot(num_dp, data[:, s], color=model_colors[s], marker=markers[s], linewidth=lw,label=score_list[s],
    #                    markersize=6)
    #     # Plot expected posterior density (cross entropy)
    #     ax[m].plot(num_dp, np.abs(post_density[m, :]), color=model_colors[4], marker=markers[4], linewidth=1,
    #                   label="CE", markersize=6)
    #
    #     handles, labels = ax[m].get_legend_handles_labels()
    #
    #     # Properties
    #     ax[m].set_ylabel('Score values')
    #     ax[m].grid(b=True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
    #     ax[m].set_title(subplot_titles[m], loc='left', fontweight='bold')
    #
    # ax[m].set_xlabel("Number of data points")
    #
    # # Legend:
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1.0))
    # # Figure properties
    # plt.subplots_adjust(top=0.95, bottom=0.1, wspace=0.1, hspace=0.3)
    # plt.margins(y=0.5, tight=True)
    # plt.show(block=False)


# BMS Model Results: -------------------------------------------------------------------------
def plot_outputs(model_list, save_name):
    """
    Function plots the prior and posterior model outputs for each model in a single plot (each plot is a subplot)

    Args:
    -----------------------
    :param model_list: list with instances of Bayes_Inference classes
    :param save_name: file path and name with which to save the plot
    """
    # Get value limits:
    stacked = model_list[0].output[:, 0]
    for i in range(1, len(model_list)):
        stacked = np.hstack((stacked, model_list[i].output[:, 0]))
    hist, bins = np.histogram(stacked, bins=50)

    bins2 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    # Fig size
    fig, ax = plt.subplots(figsize=plot_size(fraction=1, rows=len(model_list)))
    i = 1
    for m, model in enumerate(model_list):
        ax = plt.subplot(len(model_list), 2, i)
        # ax.hist(model.output[:, 0], bins=bins, color=model_colors[m])
        ax.hist(model.output[:, 0], color=model_colors[0], density=True)

        ax1 = plt.subplot(len(model_list), 2, i + 1)
        ax1.hist(model.post_output[:, 0], bins=bins2, color=model_colors[1], density=True)
        ax1.set_ylim(0, 1)

        # Set title
        if m == 0:
            ax.set_title("Prior Model Output", y=1.2)
            ax1.set_title("Posterior Model Output", y=1.2)
        # Set x axis label
        if m == len(model_list) - 1:
            ax.set_xlabel("Model output")
            ax1.set_xlabel("Model output")
        # Set y-label
        ax.set_ylabel("p(x)")

        ax.set_title(subplot_titles[i - 1] + " " + model.model_name, loc='left', fontweight='bold')
        ax1.set_title(subplot_titles[i] + model.model_name, loc='left', fontweight='bold')

        t = ax.yaxis.get_offset_text()
        t.set_x(-0.2)

        i += 2
    # fig.suptitle("Model Outputs")
    # Plot config
    plt.subplots_adjust(top=0.92, bottom=0.1, wspace=0.25, hspace=0.5)
    plt.margins(y=1, tight=True)
    # Save
    plt.savefig(save_name)

    plt.show(block=False)


def plot_prior_post(model_list, save_name, share=True):
    """
        Function plots the prior distribution and the posterior distribution for a given number of parameters for each
        model being analyzed. Here, each model has a different prior distribution.

        Args
        --------------------------------
        :param model_list: list with instances of Bayes_Inference classes
        :param save_name: file path in which to save the plot
        :param share: boolean, if True the different plots share x,y axis and the density of each value is graphed in
        the histogram.

        Notes:
        -------------
        * Function is for models in which all parameters have the same prior. The prior for one parameter (any) is
         plotted with a dotted line, and the posterior for a given number of parameters with a solid line.
        """
    # Get number of rows and columns for the
    rows = len(model_list)
    col = 1
    num_params = 3
    if share:
        share_axis = True
        dens = True
    else:
        share_axis = False
        dens = False

    fig, ax = plt.subplots(rows, col, figsize=plot_size(rows=rows), sharey=share_axis, sharex=share_axis)

    # Plot posteriors for each model:
    for m, model in enumerate(model_list):  # Each model (row)
        for p in range(0, num_params):  # Plot parameters
            # Find y range:
            if p < model.posterior.shape[1]:
                # Plot
                if p == 0:
                    ax[m].hist(model.prior[:, p], bins=10, histtype='step', color='black', linestyle='dashed',
                               label='Prior', linewidth=2, density=True)

                ax[m].hist(model.posterior[:, p], bins=10, histtype='step', color=param_colors[p],
                           label=f'$\omega_{str(p + 1)}$', linewidth=1, density=dens)
            else:
                continue

        # Subplot config
        ax[m].set_ylabel("p(x)")
        ax[m].set_ylim(0, 0.75)  # Change if limits change

        ax[m].set_title(subplot_titles[m] + "  " + model.model_name, loc='left', fontweight='bold')
        handles, labels = ax[m].get_legend_handles_labels()

    # Plot config
    ax[m].set_xlabel("Parameter value")
    plt.subplots_adjust(top=0.92, bottom=0.15, wspace=0.25, hspace=0.45)
    fig.legend(handles, labels, loc='lower center', title="Parameter", ncol=num_params+1)
    plt.margins(y=1, tight=True)

    # Save
    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


def plot_likelihoods(model_list, name, path):
    """
    Function plots the likelihoods for different parameter 1 and parameter 2 combinations (from the prior) as a scatter
    plot, with plot marker colors set by the likelihood values.

    Args:
    ------------------------------
    :param model_list: list with instances of Bayes_Inference classes
    :param name: name with which to save the plot
    :param path: folder path in which to save the plot
    """
    # model = model_list[0]
    row = len(model_list)
    col = 2
    fig, ax = plt.subplots(figsize=plot_size(rows=row))
    i = 1
    for m, model in enumerate(model_list):
        # Values
        x_prior = model.prior[:, 0]
        y_prior = model.prior[:, 1]
        z_prior = model.likelihood
        x_post = model.posterior[:, 0]
        y_post = model.posterior[:, 1]
        z_post = model.post_likelihood

        # Set axis
        ax = plt.subplot(row, col, i)
        ax1 = plt.subplot(row, col, i + 1)
        # Set axis limits:
        vmax, vmin = np.max(model.likelihood), np.min(model.likelihood)

        # Plot
        ax.scatter(x_prior, y_prior, c=z_prior, s=0.05, vmax=vmax, vmin=vmin, cmap='coolwarm', marker='o')
        im = ax1.scatter(x_post, y_post, c=z_post, s=0.05, vmax=vmax, vmin=vmin, cmap='coolwarm', marker='o')
        # # Color bar
        # cbar = fig.colorbar(im, ax=ax1)
        # cbar.set_label("Likelihood", labelpad=-2)

        # Labels
        ax.set_xlabel('$\omega_1$')
        ax.set_ylabel('$\omega_2$')
        ax1.set_xlabel('$\omega_1$')
        ax1.set_ylabel('$\omega_2$')
        ax.set_xlim(np.min(y_prior) - 1, np.max(y_prior) + 1)
        ax.set_ylim(np.min(y_prior) - 1, np.max(y_prior) + 1)
        ax1.set_xlim(np.min(y_prior) - 1, np.max(y_prior) + 1)
        ax1.set_ylim(np.min(y_prior) - 1, np.max(y_prior) + 1)

        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label("Likelihood", labelpad=10)

        # Titles
        if m == 0:
            ax.set_title("Likelihoods", y=1.15)
            ax1.set_title("Posterior Likelihoods", y=1.15)
        ax.set_title(subplot_titles[i - 1], loc='left', fontweight='bold')
        ax1.set_title(subplot_titles[i], loc='left', fontweight='bold')

        i += 2

    # Plot config
    plt.subplots_adjust(top=0.93, bottom=0.1, wspace=0.25, hspace=0.55)
    plt.margins(y=1, tight=True)
    # Save
    plt.savefig(path / f'{name}.png')

    plt.show(block=False)

    # row = 5
    # col = 2
    # fig, ax = plt.subplots(row, col, figsize=plot_size(height_fraction=1))
    #
    # for m, model in enumerate(model_list):
    #     # Values
    #     x_prior = model.prior[:, 0]
    #     y_prior = model.prior[:, 1]
    #     z_prior = model.likelihood
    #     x_post = model.posterior[:, 0]
    #     y_post = model.posterior[:, 1]
    #     z_post = model.post_likelihood
    #
    #     # Set axis
    #     # ax = plt.subplot(row, col, i)
    #     # ax1 = plt.subplot(row, col, i + 1)
    #     # Set axis limits:
    #     vmax, vmin = np.max(model.likelihood), np.min(model.likelihood)
    #
    #     # Plot
    #     ax[m, 0].scatter(x_prior, y_prior, c=z_prior, s=0.05, vmax=vmax, vmin=vmin, cmap='coolwarm')
    #     im = ax[m, 1].scatter(x_post, y_post, c=z_post, s=0.05, vmax=vmax, vmin=vmin, cmap='coolwarm')
    #     # # Color bar
    #     # cbar = fig.colorbar(im, ax=ax1)
    #     # cbar.set_label("Likelihood", labelpad=-2)
    #
    #     # Labels
    #     ax[m, 0].set_xlabel('$\omega_1$')
    #     ax[m, 0].set_ylabel('$\omega_2$')
    #     ax[m, 1].set_xlabel('$\omega_1$')
    #     ax[m, 1].set_ylabel('$\omega_2$')
    #
    # # Plot config
    # # Color bar
    # cb_ax = fig.add_axes([0.95, 0.1, 0.5, 0.8])
    # cbar = fig.colorbar(im, ax=cb_ax, location="right", pad=0.5)
    # cbar.set_label("Likelihood", labelpad=-2)
    # plt.subplots_adjust(top=0.95, bottom=0.1, wspace=0.15, hspace=0.65)
    # plt.margins(y=1, tight=True)
    # # Save
    # # plt.savefig(path + "\\" + name)
    #
    # plt.show()


# BMJ --------------------------------------------------------------------------------
def plot_confusion_matrix(data_list, model_names, score_name, save_name, use_labels=True, num_dp=None,
                          compat_models=True):
    """
    Function plots the confusion matrix/matrices for a given score. There is one subplot for each number of data points
     analyzed.

    Args:
    -----------------------------
    :param data_list: list with np.arrays, one for each number of data points analyzed
    :param model_names: list with model names, to be placed on both axis of confusion matrix
    :param save_name:  file path where to save results
    :param score_name: string with name of score (elpd, RE, IE)
    :param use_labels: boolean, if True uses labels from the model_names list, if False assigns numbers to x and
    y labels
    :param num_dp: list, with number of data points with which each np.array in 'data_list' was calculated. If none is
    given, then the number of data points will not be added to each plot title
    :param compat_models: boolean, True if models are comparable and np.nan means 'inf' values, False if models
    are not compatible and this np.nan means the calculation was not done.

    Note: each column of each confusion matrix is plotted individually, and with a different color.
    """

    # Plot properties
    cm = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'bone', 'winter']
    if score_name == "IE" or score_name == "NNCE":
        cm = ['Blues_r', 'Oranges_r', 'Greens_r', 'Reds_r', 'Purples_r', 'bone_r', 'winter_r']

    if compat_models:
        if score_name == "IE" or score_name == 'NNCE':
            null_val = "inf"
        elif score_name == "BME":
            null_val = "-"
        else:
            null_val = "-inf"
    else:
        null_val = "-"

    if len(data_list) == 1:
        sq = False
        size_text = SMALL_SIZE
    else:
        sq = True
        size_text = SMALL_SIZE - 1

    # Model names as numbers only
    if not use_labels:
        model_names = np.arange(1, len(model_names) + 1)
        tick_size = SMALL_SIZE
    else:
        tick_size = SMALL_SIZE - 1

    fig = plt.figure(figsize=plot_size())
    outer = gridspec.GridSpec(1, len(data_list))

    for o in range(0, len(data_list)):
        df = pd.DataFrame(data_list[o], index=model_names, columns=model_names)
        if score_name == "ELPD":
            df = df * -1

        inner = gridspec.GridSpecFromSubplotSpec(1, df.columns.size, subplot_spec=outer[0, o], wspace=0.0, hspace=0.1)

        # common X label:
        ax = plt.Subplot(fig, outer[o])
        ax.axis('off')

        fig.add_subplot(ax)

        for i, (s, c) in enumerate(zip(df.columns, cm)):
            ax = plt.Subplot(fig, inner[i])
            # Get limits:
            if score_name == 'BME':
                v_max = 1
                v_min = 0
            else:
                v_max = np.nanmax(np.array([df[s].values])) + 0.5
                v_min = np.nanmin(np.array([df[s].values])) - 0.5
            # Plot null values
            v_nu = np.where(df[s].isna(), 0, np.nan).reshape(len(model_names), 1)
            lab = np.full_like(df[s], null_val, dtype=object).reshape(len(model_names), 1)
            sn.heatmap(v_nu, ax=ax, cbar=False, annot=lab, fmt="",
                       annot_kws={"size": size_text, "va": "center_baseline", "color": "black"},
                       cmap=ListedColormap(['none']), square=sq)
            # Plot values
            sn.heatmap(np.array([df[s].values]).T, yticklabels=df.index, xticklabels=[s], annot=True, fmt='.2f', ax=ax,
                       cmap=c, cbar=False, annot_kws={"size": size_text, "color": "black"}, vmin=v_min, vmax=v_max,
                       square=sq, linewidths=0.01, linecolor='black')
            # ax.set_aspect("equal")
            # ticks
            ax.tick_params(labelsize=tick_size)
            ax.xaxis.tick_top()  # x axis on top
            ax.xaxis.set_label_position('top')

            # y-axis
            if i > 0 or o > 0:
                ax.yaxis.set_ticks([])
            else:
                ax.set_ylabel(score_name + " for model", labelpad=10)

            # X axis
            if i == (math.ceil(len(df.columns) / 2) - 1):
                ax.set_xlabel('Data generating model', loc='center', labelpad=10)
            # Title:
            if i == 0 and len(data_list) > 1:
                if num_dp:
                    ax.set_title(subplot_titles[o] + " N=" + str(num_dp[o]), loc='left', fontweight='bold')
                else:
                    ax.set_title(subplot_titles[o], loc='left', fontweight='bold')

            fig.add_subplot(ax)

    plt.subplots_adjust(top=0.80, bottom=0.1, wspace=0.25, hspace=1.5)
    plt.margins(y=1, tight=True)

    # Save
    plt.savefig(save_name)
    plt.show(block=False)
    x=1


def plot_confusion_matrix_all_scores(data_list, model_names, save_name, uncertainty, data_type, compat_models=True,
                                     use_labels=True):
    """
    Function plots the confusion matrix/matrices for all scores, given a number of data points. There is one subplot
    for each score.

    Args:
    -----------------------------
    :param data_list: list with np.arrays, one for each score, in the order (BME, ELPD, RE, IE)
    :param model_names: list with model names, to be placed on both axis of confusion matrix
    :param save_name:  file path where to save results
    :param uncertainty: int, 1 if data_list contains total score values, 2 if it contains weights, 3 if it contains
    normalized values
    :param compat_models: boolean, True if models are comparable and np.nan means 'inf' values, False if models
    are not compatible and this np.nan means the calculation was not done.
    :param use_labels: boolean, if True uses labels from the model_names list, if False assigns numbers to x and
    y labels

    Note: each column of each confusion matrix is plotted individually, and with a different color.
    """
    # Model names as numbers only
    if not use_labels:
        model_names = np.arange(1, len(model_names) + 1)
        tick_size = SMALL_SIZE
    else:
        tick_size = SMALL_SIZE - 2

    # Check if bme is in weights or log(BME)
    if np.max(data_list[0]) <= 1 and np.min(data_list[0]) > 0:
        bme_weights = True
    else:
        bme_weights = False

    # CM labels:
    text_format = '.2f'
    text_size = SMALL_SIZE - 1
    for i in range(0, len(data_list)):
        if np.any(data_list[i] >= 1000) or np.any(data_list[i] <= 0.0001):
            text_format = '.2g'
            text_size = SMALL_SIZE - 3
            break

    fig = plt.figure(figsize=plot_size())
    outer = gridspec.GridSpec(2, 2)
    rw = 0
    cl = 0
    for o in range(0, len(data_list)):  # each score
        # Determine color scheme
        if data_type == 3:  # for normalized data
            cm = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'bone', 'winter']
        else:
            if o == 3 or (o == 1 and uncertainty) or (o == 0 and not bme_weights and uncertainty):  # Neg. value better
                cm = ['Blues_r', 'Oranges_r', 'Greens_r', 'Reds_r', 'Purples_r', 'bone_r', 'winter_r']
            else:  # Larger value is better
                cm = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'bone', 'winter']

        # determine null value:
        if data_type == 2 or data_type == 3 or not compat_models:
            null_val = "-"
        else:
            if uncertainty or o == 2 or o == 3:
                null_val = "inf"
            else:
                null_val = "-inf"

        df = pd.DataFrame(data_list[o], index=model_names, columns=model_names)
        if data_type == 1:
            if (o == 1 and not uncertainty) or (o == 0 and not uncertainty and not bme_weights):
                df = df * -1

        inner = gridspec.GridSpecFromSubplotSpec(1, df.columns.size, subplot_spec=outer[rw, cl], wspace=0.0, hspace=0.1)

        # common X label:
        ax = plt.Subplot(fig, outer[o])
        ax.axis('off')
        fig.add_subplot(ax)

        for i, (s, c) in enumerate(zip(df.columns, cm)):
            ax = plt.Subplot(fig, inner[i])
            # Get limits:
            if data_type == 1:
                if o == 0 and bme_weights:
                    v_max = 1
                    v_min = 0
                else:
                    v_max = np.nanmax(np.array([df[s].values])) + 0.5
                    v_min = np.nanmin(np.array([df[s].values])) - 0.5
            elif data_type == 2:
                v_max = 1
                v_min = 0
            else:
                v_max = np.nanmax(np.array([df[s].values])) + 0.1
                v_min = np.nanmin(np.array([df[s].values])) - 0.1
            v_nu = np.where(df[s].isna(), 0, np.nan).reshape(len(model_names), 1)
            lab = np.full_like(df[s], null_val, dtype=object).reshape(len(model_names), 1)
            sn.heatmap(v_nu, ax=ax, cbar=False, annot=lab, fmt="",
                       annot_kws={"size": text_size, "va": "center_baseline", "color": "black"},
                       cmap=ListedColormap(['none']), square=False)
            # Plot values
            sn.heatmap(np.array([df[s].values]).T, yticklabels=df.index, xticklabels=[s], annot=True, fmt=text_format,
                       ax=ax, cmap=c, cbar=False, annot_kws={"size": text_size}, vmin=v_min,
                       vmax=v_max, square=False, linewidths=0.01, linecolor='black')  # fmt='.2f'
            # annot_kws = {"size": text_size, "color": "black"}
            # ax.set_aspect("equal")

            # ticks
            ax.tick_params(labelsize=tick_size)
            ax.xaxis.tick_top()  # x axis on top
            ax.xaxis.set_label_position('top')
            # y-axis
            if i > 0:
                ax.yaxis.set_ticks([])
            else:
                if o != 0:
                    if uncertainty:
                        lb = score_list[o] + " for model"
                    else:
                        lb = score_list_2[o] + " for model"
                else:
                    if bme_weights:
                        lb = "BME weight for model"
                    else:
                        if uncertainty:
                            lb = score_list[o] + " for model"
                        else:
                            lb = score_list_2[o] + " for model"

                ax.set_ylabel(lb, labelpad=10)

            # X axis
            if i == (math.ceil(len(df.columns) / 2) - 1):
                ax.set_xlabel('Data generating model', loc='center')
            # Title:
            if i == 0 and len(data_list) > 1:
                ax.set_title(subplot_titles[o], loc='left', fontweight='bold')

            fig.add_subplot(ax)

        cl = cl + 1
        if o == 1:
            rw = rw + 1
            cl = 0

    plt.subplots_adjust(top=0.85, bottom=0.1, wspace=0.45, hspace=0.55)
    plt.margins(y=1, tight=True)
    # Save

    plt.savefig(save_name)
    plt.show(block=False)
    x = 1


def plot_bmj_diagonal(list_scores, num_dp, model_names, path):
    """
    Function plots the diagonal values for each score with increasing number of data points. It first extracts the
    diagonal values from the input lists and then calls 'plot_evolving_values' to plot the results for each score in a
    subplot.

    Args:
    -----------------------------------------
    :param list_scores: list, with lists. Each sub-list contains 'N' number of confusion matrix arrays, one for each
    number of data points.
    :param num_dp: list of int, with number od data points analyzed in each run
    :param model_names: list of strings with the model names
    :param path: path of folder to save the results to
    """
    results_list = np.full((len(num_dp), len(model_names), 4), 0.0)
    for s, s_array in enumerate(list_scores):  # loop through each score (col in each array)
        for n, n_array in enumerate(s_array):  # loop through each number of data points (array in 3D array)
            results_list[n, :, s] = np.diagonal(n_array)

    save_name = path + "AM_EvolvingDiagonal_BMJ.eps"
    plot_evolving_values(results_list, num_dp, model_names, save_name)


