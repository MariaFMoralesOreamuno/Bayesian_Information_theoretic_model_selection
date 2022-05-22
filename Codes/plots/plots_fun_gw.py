"""
Author: María Fernanda Morales Oreamuno

Created: 19/11/2021
Last update: 08/12/2021

Module contains the functions to plot and save different results specifically for the groundwater models, as part of
the thesis "Bayesian and Information-Theoretic scores for Model Similarity Analysis".
"""

from plots.plots_fun import *
from config import *

ext = ".pdf"


# General plotting functions
def plot_bar(axis, x, y, index, uncertainty, value_label=False, isolate=False):
    """
    Function plots bar graphs for each score, individually and returns it to main function.

    Args:
    :param axis: plot axis
    :param x: list, x value labels (e.g. model names)
    :param y: np.array, y values (score values for each model)
    :param index: int, to get y axis label
    :param uncertainty: boolean, when True uses score names with entropy, when False uses predictive capability related
    labels
    :param value_label: boolean, when True, adds label to each bar
    :param isolate: boolean, true if a border should be added to the flow model(s)

    Note: User has the option to add value labels over each bar.
    Note: The flow model is singled out (a border is added) to highlight that this model can't be compared for a given
    score.
    """
    # Generate bar plot:
    x_loc = np.arange(1, len(x) + 1)

    # get colors:
    # edge_colors = []
    # line_w = []
    # for i in range(0, len(x)):
    #     if 'f' in x[i] and isolate:  # Only flow model has "f" in model name
    #         edge_colors.append('red')
    #         line_w.append(2)
    #     else:
    #         edge_colors.append('none')
    #         line_w.append(0)
    # bar_plot = axis.bar(x_loc, y, width=width, color=model_colors[index], edgecolor=edge_colors, linewidth=line_w)

    for i in range(0, len(x)):
        if 'f' in x[i] and isolate:  # Only flow model has "f" in model name
            tr = 0.25
        else:
            tr = 1
        bar_plot = axis.bar(x_loc[i], y[i], width=width, color=model_colors[index], alpha=tr)

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

    # x-Ticks: set one tick for each bar
    axis.set_xticks(np.arange(1, len(x) + 1))
    axis.set_xticklabels(x)

    if value_label:
        add_label(bar_plot)

    return axis


# BMS
def plot_scores_bar_gw(score_array, model_names, save_name, uncertainty):
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
    Note: the flow model is singled out for BME and NNCE values, to highlight the inadequacy of said scores in this case
    """

    # plt.figure(1, constrained_layout=True)
    plt.figure(figsize=plot_size())

    # Loop through each score
    for i in range(0, score_array.shape[1]):
        p = plt.subplot(2, 2, i + 1)
        # Extract score data
        data = score_array[:, i]
        # if data is not to be in uncertainty
        if not uncertainty and (i == 0 or i == 1):
            data = -1 * data

        # Determine whether to isolate flow model:
        if i < 2:
            isolate = True
        else:
            isolate = False

        p = plot_bar(p, model_names, data, i, uncertainty, isolate=isolate)
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


def plot_individual_gw_scores(score_array, model_names, save_name, uncertainty):
    """
        Function plots each BMS score for each model as individual bar graphs. 

        Args:
        ----------
        :param score_array: array with scores [MxS] where S=number of scores (4) and M = number of models. The score order
        is: BME, NNCE, RE, IE
        :param model_names: Array with shape [1xM] with model names, as strings
        :param save_name: file path where to save the resulting figure
        :param uncertainty: boolean, if true it plots -log(BME) and NNCE, if false it plots BME and ELPD
        ----------
        :return: Save plots (no return)

        Note: function calls another function to generate the bar graphs in each loop.
        Note: the flow model is singled out for BME and NNCE values, to highlight the inadequacy of said scores in this
        case
        """
    # Loop through each score
    for i in range(0, score_array.shape[1]):
        fig, p = plt.subplots(1, figsize=plot_size(fraction=0.75))  # Each plot will be half a LaTex page

        # Extract score data
        # If geostatistical model is removed:
        if i == score_array.shape[1]-1 and score_array[score_array.shape[0]-1, i] == 0:
            data = score_array[:-1, i]
            mn = model_names[:-1]
        else:
            data = score_array[:, i]
            mn = model_names

        # if data is not to be in uncertainty
        if not uncertainty and (i == 0 or i == 1):
            data = -1 * data

        # Determine whether to isolate flow model:
        if i < 2:
            isolate = True
        else:
            isolate = False

        p = plot_bar(p, mn, data, i, uncertainty, isolate=isolate)
        # if i < score_array.shape[1] / 2:
        #     p.set_xlabel("")
        p.set_title(subplot_titles[i], loc='left', fontweight='bold')
        p.grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
        # p.text(1, 1, subplot_titles[i])

        # Format:
        plt.subplots_adjust(top=0.92, bottom=0.15, wspace=0.45, hspace=0.3)
        plt.margins(y=0.5, tight=True)

        # Save plot:
        if uncertainty:
            sn = os.path.join(f'{save_name}_{score_list[i]}' + ext)
        else:
            sn = os.path.join(f'{save_name}_{score_list_2[i]}' + ext)
        plt.savefig(sn)

        # Print
        plt.show(block=False)


def plot_stacked_score_calculation_gw(data, ce, labels, save_name):
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

    #   Stacked plot:
    for i in range(0, data.shape[0]):
        # Top plot ------------------------------------------------------------------------------------------ #
        ax[0].bar(x[i] + loc[0], data[i, 0], label=score_list[0], color=model_colors[0], width=width / num_bars)  # BME
        ax[0].bar(x[i] + loc[1], data[i, 1], label='-ELPD', color=model_colors[1], width=width / num_bars)  # -elpd
        if data[i, 0] < 0 and data[i, 1] < 0:
            ax[0].bar(x[i] + loc[0], -data[i, 2], label=score_list[2], color=model_colors[2], width=width / num_bars,
                      bottom=data[i, 0])  # RE
        else:
            ax[0].bar(x[i] + loc[1], data[i, 2], label=score_list[2], color=model_colors[2], width=width / num_bars,
                      bottom=data[i, 1])  # RE

        # Bottom plot: ------------------------------------------------------------------------------------------ #
        ax[1].bar(x[i] + loc[0], -ce[i, 0], label='CE', color=model_colors[4], width=width / num_bars)  # CE
        ax[1].bar(x[i] + loc[1], data[i, 3], label=score_list[3], color=model_colors[3], width=width / num_bars)

        if -ce[i, 0] < 0 and data[i, 3] < 0:
            ax[1].bar(x[i] + loc[0], -data[i, 2], label=score_list[2], color=model_colors[2], width=width / num_bars,
                      bottom=-ce[i, 0])  # RE
        else:
            ax[1].bar(x[i] + loc[1], data[i, 2], label=score_list[2], color=model_colors[2], width=width / num_bars,
                      bottom=data[i, 3])  # RE

    # Top plot properties ----------------------------------------------------------------------------------- #
    # get limits
    max_0, min_0 = get_limits([data[:, 0], data[:, 1], data[:, 2]])
    max_0, min_0 = get_plot_range(max_0, min_0)

    ax[0].set_ylabel('Scores')
    ax[0].set_title(subplot_titles[0] + ": RE = -log(BME) - ELPD", loc='left', fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylim(min_0, max_0)

    # Bottom plot: ------------------------------------------------------------------------------------------ #
    # get limits:
    max_1, min_1 = get_limits([data[:, 2], -ce[:, 0], data[:, 3]])
    max_1, min_1 = get_plot_range(max_1, min_1)

    ax[1].set_ylabel('Scores')
    ax[1].set_title(subplot_titles[1] + ": IE = CE - RE", loc='left', fontweight='bold')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_xlabel('Model')
    ax[1].set_ylim(min_1, max_1)

    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    by_label_1 = dict(zip(labels, handles))

    handles, labels = ax[1].get_legend_handles_labels()
    by_label_2 = dict(zip(labels, handles))

    all_labels = by_label_1.copy()
    all_labels.update(by_label_2)

    fig.legend(all_labels.values(), all_labels.keys(), loc='lower center', ncol=5)

    ax[0].grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
    ax[1].grid(b=True, which='major', axis='y', color='lightgrey', linestyle='--')
    plt.subplots_adjust(top=0.92, bottom=0.2, wspace=0.1, hspace=0.4)
    plt.margins(y=0.5, tight=True)

    # Save plot:
    plt.savefig(save_name)

    plt.show(block=False)


def plot_bme_weights_gw(data, x, stack_labels, save_name):
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
    # ax.set_xlabel("Groundwater model setup")
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


def plot_prior_post_gw(model_list, save_name, share=True):
    """
        Function plots the prior and the posterior distribution for a given number of parameters for each model being
        analyzed. Both are plotting in the same subplot.

        Args
        --------------------------------
        :param model_list: list with instances of Bayes_Inference classes
        :param save_name: file path in which to save the plot
        :param share: boolean, if True the different plots share x,y axis and the density of each value is graphed in
        the histogram.

        Notes:
        -------------
        * The prior will be plotted with a dotted black line and the posterior in different colors.
        """
    # Get number of rows and columns for the
    rows = len(model_list)
    col = 1
    num_params = 3
    if share:
        share_axis = True
    else:
        share_axis = False

    fig, ax = plt.subplots(rows, col, figsize=plot_size(rows=rows), sharey=share_axis, sharex=share_axis)

    if ax.ndim == 1 and rows == 1:
        ax = np.reshape(ax, (1, col))

    # Plot posteriors for each model:
    i = 0
    for m, model in enumerate(model_list):  # Each model (row)
        ax[m].hist(model.prior[:, 0], bins=10, histtype='step', color='black', label='prior', linewidth=2,
                   density=True, linestyle='dashed')
        for p in range(0, num_params):  # Plot parameters
            # Find y range:
            x = model.posterior.shape[1]
            if p < model.posterior.shape[1]:
                n, bins = np.histogram(model.posterior[:, p], bins=10)
                # Plot
                ax[m].hist(model.posterior[:, p], bins=10, histtype='step', color=param_colors[p], label=str(p + 1),
                           linewidth=2, density=True)
            else:
                continue

        # Subplot config
        ax[m].set_ylabel("Frequency")
        ax[m].set_ylim(0, 0.75)

        ax[m].set_xlim(-16, -7)

        # -- Title --
        ax[m].set_title(subplot_titles[m] + " " + model.model_name, loc='left', fontweight='bold')

        handles, labels = ax[m].get_legend_handles_labels()

    # Plot config
    ax[m].set_xlabel("log(K) values")

    plt.subplots_adjust(top=0.92, bottom=0.15, wspace=0.25, hspace=0.55)
    fig.legend(handles, labels, loc='lower center', title="Parameter", ncol=num_params+1)
    plt.margins(y=1, tight=True)

    # Save
    plt.savefig(save_name)

    plt.show(block=False)
    stop = 1


def plot_outputs_gw(model_list, save_name, m_type, dp=0):
    """
    Function plots the prior and posterior model outputs for each model, for a given type and measurement point for
    the groundwater model (Each subplot corresponds to a different model, and both prior and posterior are plotted in
    the same subplot)

    Args:
    -----------------------
    :param model_list: list with instances of Bayes_Inference classes
    :param save_name: file path and name with which to save the plot
    :param dp: which data point to plot
    :param m_type: string
    """
    # Set number of rows (subplots)
    cols = 1
    rows = 0
    for m, model in enumerate(model_list):
        if model.output.shape[1] > dp:
            rows = rows + 1

    if m_type == 'Y  ':
        bins = np.arange(-15, -6, 1)
        lim = [-15, -7]
    else:
        bins = np.arange(0, 1.2, 0.2)
        lim = [0, 1]

    # Fig size
    fig, ax = plt.subplots(rows, cols, figsize=plot_size(rows=len(model_list)))
    r = 0
    for m, model in enumerate(model_list):
        m = r
        if model.output.shape[1] > dp:
            weights = np.ones_like(model.output[:, dp]) / len(model.output[:, dp])
            ax[m].hist(model.output[:, dp], color=model_colors[0], density=True, label='Prior', weights=weights)

            weights = np.ones_like(model.post_output[:, dp]) / len(model.post_output[:, dp])
            ax[m].hist(model.post_output[:, dp], color='black', density=True, alpha=0.75, label='Posterior',
                       weights=weights)

            # lines
            avg = model.measurement_data.meas_values[0, dp]
            err = model.measurement_data.error[0, dp]
            hist_1, bins = np.histogram(model.output[:, dp], bins=len(bins), density=True)
            hist_2, bins = np.histogram(model.post_output[:, dp], bins=len(bins), density=True)
            mx = np.max(hist_1)
            mx2 = np.max(hist_2)
            max_tot = np.max(np.array([mx, mx2]))

            ax[m].plot([avg - err, avg - err], [0, max_tot], 'r-')
            ax[m].plot([avg + err, avg + err], [0, max_tot], 'r-', label='$y_{o,i} \pm \epsilon_i$')

            r = r+1  # To loop through each model, but only plot the ones that contain data

        # Set title
        if m == 0:
            ax[m].set_title("Model Output", y=1.2)
        ax[m].set_ylabel("p(x)")

        # Set x axis label
        if m == len(model_list) - 1:
            ax[m].set_xlabel(m_type.replace(" ", "") + " model output")

        ax[m].set_title(subplot_titles[m] + " " + model.model_name, loc='left', fontweight='bold')

        ax[m].set_xlim(lim)
        t = ax[m].yaxis.get_offset_text()
        t.set_x(-0.1)

        if lim[0] < (avg - err) and lim[1] > (avg + err):
            ax[m].set_xlim(lim)
        else:
            if lim[0] > (avg - err) and lim[1] > (avg + err):
                ax[m].set_xlim((avg - err - 0.1), lim[1])
            elif lim[0] < (avg - err) and lim[1] < (avg + err):
                ax[m].set_xlim(lim[0], (avg + err + 0.1))
            else:
                ax[m].set_xlim((avg + err - 1), (avg + err + 0.1))

    # Plot config
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.subplots_adjust(top=0.92, bottom=0.1, wspace=0.25, hspace=0.5)
    plt.margins(y=1, tight=True)

    # Save
    plt.savefig(save_name)
    plt.show(block=False)
    stop = 1


def plot_likelihoods_gw(model_list, save_name):
    """
    Function plots the prior and posterior log(likelihoods) for each competing model in the model selection analysis.

    Args:
    ------------------------------------
    :param model_list: list, with BayesInference instances, one for each competing model
    :param save_name: str, path and name.ext with which to save the resulting plots.

    Note: if a likelihood is equal to zero, and thus the log cannot be calculated, the function assigns it a value of
    1e-300 in order to allow for the calculation of the value.
    """
    rows = len(model_list)
    cols = 1
    fig, ax = plt.subplots(rows, cols, figsize=plot_size(rows=len(model_list)))

    for m, model in enumerate(model_list):
        plot_y = np.where(model.likelihood == 0, 1e-300, model.likelihood)
        plot_y = np.log(plot_y)

        ax[m].hist(plot_y, color=model_colors[0], density=True, bins=50,label='prior')
        ax[m].hist(np.log(model.post_likelihood), color='black', density=True, bins=50, alpha=0.75, label='Posterior')

        hist_1, bins = np.histogram(np.log(model.post_likelihood), bins=50, density=True)
        max_y = np.max(hist_1)
        ax[m].plot([np.max(plot_y), np.max(plot_y)], [0, max_y], color='red', linewidth=2, label='Max value')
        # ax[m].set_xscale('log')
        t = ax[m].yaxis.get_offset_text()
        t.set_x(-0.1)

        ax[m].set_ylabel("Frequency")
        if m == len(model_list)-1:
            ax[m].set_xlabel("ln(likelihood)")
        ax[m].set_title(subplot_titles[m] + " " + model.model_name, loc='left', fontweight='bold')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)

    plt.subplots_adjust(top=0.92, bottom=0.1, wspace=0.25, hspace=0.5)
    plt.margins(y=1, tight=True)

    plt.savefig(save_name)
    plt.show(block=False)
    stop=1


# BMJ
def plot_outputs_bmj_gw(mj_list, model_list, save_name, m_type, dp):
    """
    Function plots the prior and posterior model outputs for each model and for a given type and measurement point for
    the groundwater model (Each subplot corresponds to a different model)

    Args:
    -----------------------
    :param mj_list: list with instances of Measurement_Data class, with synthetically true runs for BMJ
    :param model_list: list with instances of Bayes_Inference classes
    :param save_name: file path and name with which to save the plot
    :param dp: which data point to plot
    :param m_type: string
    """

    rows = len(model_list)
    cols = 1

    if m_type == 'Y  ':
        bins = np.arange(-15, -6, 1)
        lims = [-15, -7]
    else:
        bins = np.arange(0, 1.2, 0.2)
        lims = [0, 1]

    # Fig size
    fig, ax = plt.subplots(rows, cols, figsize=plot_size(rows=len(model_list)))
    for m, model in enumerate(model_list):
        syn_model = mj_list[m]
        if model.output.shape[1] > dp:
            ax[m].hist(model.output[:, dp], color=model_colors[m], density=True)

            ax[m].hist(syn_model.meas_values[:, dp], histtype='step', color='black', linestyle='dashed',
                       density=True)

        # Set title
        if m == len(model_list) - 1:
            ax[m].set_xlabel(m_type.replace(" ", "") + " model output")

        ax[m].set_title(subplot_titles[m] + " " + model.model_name, loc='left', fontweight='bold')
        ax[m].set_xlim(lims)

    # Plot config
    plt.subplots_adjust(top=0.92, bottom=0.1, wspace=0.25, hspace=0.5)
    plt.margins(y=1, tight=True)

    # Save
    # plt.savefig(save_name)

    plt.show(block=False)
    stop = 1


# Models
def plot_zones(path_k, zones, point_loc, save_name):
    """
    Function plots the synthetic true log(K) distribution and a given zone classification that stemmed from said
    distribution.

    Args
    ---------------------------
    :param path_k: string, path where the .mat containing the true log(K) spatial distribution is located
    :param zones: np.array, with int values, corresponding to classification values for each cell in the model grid
    :param point_loc: np.array, with coordinate values for each measurement point location. Columns are [Y, X, 1]
    :param save_name: string, path and file name with which to save the resulting plot
    """
    # plot properties
    cmap = 'viridis'
    cmap2 = plt.cm.get_cmap(cmap, len(np.unique(zones)))  # 11 discrete colors

    # Read real k values
    k_array = loadmat(path_k)['Y_true']

    # Get unique point locations:
    loc = np.unique(point_loc, axis=0)

    # get x, y, location of points
    x_loc = np.full((k_array.shape[0], k_array.shape[1]), 0.0)
    y_loc = np.full((k_array.shape[0], k_array.shape[1]), 0.0)

    loc_vals = np.arange(0, k_array.shape[0], 1)
    for i in range(0, k_array.shape[0]):
        x_loc[i, :] = loc_vals
        y_loc[:, i] = loc_vals
        # y_loc[:, i] = np.flip(loc_vals)

    rows = 1
    cols = 2
    fig, ax = plt.subplots(rows, cols, figsize=plot_size(rows=rows))
    # Left plot
    im = ax[0].pcolormesh(x_loc, y_loc, k_array)
    ax[0].scatter(loc[:, 1], loc[:, 0], s=40, facecolors='k', edgecolors='w')
    cbar = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    # Right plot
    im2 = ax[1].pcolormesh(x_loc, y_loc, zones, cmap=cmap2)
    ax[1].scatter(loc[:, 1], loc[:, 0], s=40, facecolors='k', edgecolors='w')

    dx = (np.max(zones) - np.min(zones)) / len(np.unique(zones))
    cbar = fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04,
                        ticks=np.arange(np.min(zones) + dx / 2, np.max(zones) + dx / 2, dx))
    cbar.set_ticklabels(np.arange(1, np.max(zones) + 1, 1))
    # cbar.yaxis.set_ticks[np.arange(1, np.max(zones) + 1, 1)]

    ax[0].set_aspect(1.0 / ax[0].get_data_ratio() * 1)
    ax[1].set_aspect(1.0 / ax[1].get_data_ratio() * 1)

    ax[0].set_title('a) synthetic true log(K)', loc='left', fontweight='bold')
    ax[1].set_title('b) zone classification', loc='left', fontweight='bold')
    # ax[0].set_adjustable("box")

    plt.subplots_adjust(top=0.93, bottom=0.1, wspace=0.35, hspace=0.1)
    # plt.margins(y=1, tight=True)

    plt.savefig(save_name)

    plt.show(block=False)


def plot_diff_zones(path_k, zones_list, point_loc, save_name):
    """
    Function plots both zone classifications (for model zm_5 and zm_9) that stemmed from the synthetically true
    log(K) distribution.

    Args
    ---------------------------
    :param path_k: string, path where the .mat containing the true log(K) spatial distribution is located
    :param zones_list: list, with np.arrays, with int values, corresponding to classification values for each model,
    for each cell in the model grid
    :param point_loc: np.array, with coordinate values for each measurement point location. Columns are [Y, X, 1]
    :param save_name: string, path and file name with which to save the resulting plot
    """
    # plot properties
    cmap = 'viridis'

    # Read real k values
    k_array = loadmat(path_k)['Y_true']

    # Get unique point locations:
    loc = np.unique(point_loc, axis=0)

    # get x, y, location of points
    x_loc = np.full((k_array.shape[0], k_array.shape[1]), 0.0)
    y_loc = np.full((k_array.shape[0], k_array.shape[1]), 0.0)

    loc_vals = np.arange(0, k_array.shape[0], 1)
    for i in range(0, k_array.shape[0]):
        x_loc[i, :] = loc_vals
        y_loc[:, i] = loc_vals

    rows = 1
    cols = len(zones_list)
    fig, ax = plt.subplots(rows, cols, figsize=plot_size(rows=rows))

    for i in range(0, cols):

        zone_grid = zones_list[i]
        cmap_i = plt.cm.get_cmap(cmap, len(np.unique(zone_grid)))  # 11 discrete colors

        im1 = ax[i].pcolormesh(x_loc, y_loc, zone_grid, cmap=cmap_i)
        ax[i].scatter(loc[:, 1], loc[:, 0], s=40, facecolors='k', edgecolors='w')

        dx = (np.max(zone_grid) - np.min(zone_grid)) / len(np.unique(zone_grid))
        cbar = fig.colorbar(im1, ax=ax[i], fraction=0.046, pad=0.04,
                            ticks=np.arange(np.min(zone_grid) + dx / 2, np.max(zone_grid) + dx / 2, dx))
        cbar.set_ticklabels(np.arange(1, np.max(zone_grid) + 1, 1))
        # cbar.yaxis.set_ticks[np.arange(1, np.max(zones) + 1, 1)]

        ax[i].set_aspect(1.0 / ax[0].get_data_ratio() * 1)
        ax[i].set_title(f'a) zoned model {np.shape(np.unique(zone_grid))[0]}', loc='left', fontweight='bold')

    plt.subplots_adjust(top=0.93, bottom=0.1, wspace=0.35, hspace=0.1)
    # plt.margins(y=1, tight=True)

    plt.savefig(save_name)

    plt.show(block=False)


def plot_synthetic_gw(path_k, path_sol, point_loc, num_pts, save_path):
    cmap = 'viridis'

    # Read real values
    k_array = loadmat(path_k)['Y_true']
    h_array = loadmat(path_sol)['h_true']
    c_array = loadmat(path_sol)['m0_true']

    # Get unique point locations:
    loc = point_loc[0:int(num_pts), :]

    # get x, y, location of points
    x_loc = np.full((k_array.shape[0], k_array.shape[1]), 0.0)
    y_loc = np.full((k_array.shape[0], k_array.shape[1]), 0.0)

    loc_vals = np.arange(0, k_array.shape[0], 1)
    for i in range(0, k_array.shape[0]):
        x_loc[i, :] = loc_vals
        y_loc[:, i] = loc_vals

    fig, ax = plt.subplots(figsize=plot_size())

    im = ax.pcolormesh(x_loc, y_loc, k_array)
    ax.scatter(loc[:, 1], loc[:, 0], s=40, facecolors='k', edgecolors='w')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log(K)')

    labls = np.arange(1, loc.shape[0] + 1)
    for i, txt in enumerate(labls):
        ax.annotate(txt, (loc[:, 1][i] + 1, loc[:, 0][i] + 1))

    ax.set_aspect(1.0 / ax.get_data_ratio() * 1)

    save_name = os.path.join(save_path, 'synthetic_logK' + ext)
    plt.savefig(save_name)
    plt.show(block=False)

    h_array = np.reshape(h_array, (51, 51), order='F')
    c_array = np.reshape(c_array, (51, 51), order='F')

    fig, ax = plt.subplots(1, 2, figsize=plot_size())

    im = ax[0].pcolormesh(x_loc, y_loc, h_array[:-1, :-1])
    ax[0].scatter(loc[:, 1], loc[:, 0], s=40, facecolors='k', edgecolors='w')
    cbar = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    cbar.set_label('hydraulic head', labelpad=10)
    cbar.set_ticklabels(np.arange(0, 1.2, 0.2))

    # Concentration
    im = ax[1].pcolormesh(x_loc, y_loc, c_array[1:, 1:])
    ax[1].scatter(loc[:, 1], loc[:, 0], s=40, facecolors='k', edgecolors='w')
    cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label('Concentration', labelpad=10)
    cbar.set_ticklabels(np.arange(0, 1.2, 0.2))

    ax[0].set_aspect(1.0 / ax[0].get_data_ratio() * 1)
    ax[0].set_title('Hydraulic head (m)')

    ax[1].set_aspect(1.0 / ax[1].get_data_ratio() * 1)
    ax[1].set_title('Concentration')

    plt.subplots_adjust(top=0.93, bottom=0.1, wspace=0.55, hspace=0.15)

    save_name = os.path.join(save_path, 'synthetic_gw_h_c' + ext)
    plt.savefig(save_name)

    plt.show(block=False)

