"""
Module with plotting functions for the main_gw_errors.py and main_am_trans.py program, for all modules that involve
the analysis of Bayesian and information theoretic scores with increasing variable, e.g. number parameters, error value,
etc.
"""
from plots.plots_fun import *


def plot_outputs_gw_trends(model_list, save_name, m_type, dp=0):
    """
    Function plots the prior and posterior model outputs for each model and for a given type and measurement point for
    the groundwater model (Each subplot corresponds to a different model)

    Args:
    -----------------------
    :param model_list: list with instances of Bayes_Inference classes
    :param save_name: file path and name with which to save the plot
    :param dp: which data point to plot
    :param m_type: string
    """

    rows = model_list.shape[1] + 1
    cols = model_list.shape[0]

    if m_type == 'Y  ':
        bins =np.arange(-15, -6, 1)
        lims = [-15, -7]
    elif m_type == "d  ":
        bins =np.arange(-15, -6, 1)
        lims = [-3, 0]
    else:
        bins = np.arange(0, 1.2, 0.2)
        lims = [0, 1]

    fig, ax = plt.subplots(rows, cols, figsize=plot_size(rows=len(model_list)))

    i = 1
    for m in range(0, model_list.shape[0]):
        if model_list.shape[0] == 3:
            i = m+1
        else:
            i = m
        # Prior
        model = model_list[m, 0]  # Read prior from any of the model runs in row "m"
        ax[0, m].hist(model.output[:, dp], color=model_colors[i], density=False)
        ax[0, m].set_title(f"Model {model.model_name}", y=1.02)
        ax[0, m].set_xlim(lims)

        # Posterior
        for e in range(0, model_list.shape[1]):
            model = model_list[m, e]
            ax[e+1, m].hist(model.post_output[:, dp], color=model_colors[i+1], density=False)

            # lines
            avg = model.measurement_data.meas_values[0, dp]
            err = model.measurement_data.error[0, dp]
            hist_1, bins = np.histogram(model.post_output[:, dp], bins=len(bins))
            mx = np.max(hist_1)
            ax[e+1, m].plot([avg - err, avg - err], [0, mx], 'k-')
            ax[e+1, m].plot([avg + err, avg + err], [0, mx], 'k-')

            # Set title
            if e == 0:
                if m == 0:
                    ax[e+1, m].set_title(f"{subplot_titles[e+1]} Posterior")
                else:
                    ax[e + 1, m].set_title("Posterior")
            else:
                if m == 0:
                    ax[e + 1, m].set_title(subplot_titles[e + 1], loc='left', fontweight='bold')

            # Set x axis label
            if e == model_list.shape[1] - 1:
                ax[e + 1, m].set_xlabel(m_type.replace(" ", "") + " values")

            ax[e + 1, m].set_xlim(lims)

    # fig.legend(handles, labels, loc='lower center', title="Parameter", ncol=model_list.shape[1])
    plt.subplots_adjust(top=0.96, bottom=0.1, wspace=0.55, hspace=0.6)
    plt.margins(y=0.5, tight=True)

    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


def plot_changing_variable(size_array, model_name, x_values, save_name, y_label, x_label):
    """
    Function plots a variable (sample size, ESS) for increasing values, be them error factors or number of MC
    realizations

    :param size_array: np.array, of size MxN, M=number of models, N=number of loops analyzed
    :param model_name: np.array, with size M, one for each model
    :param x_values: np.array, x tick labels
    :param save_name: str, path and name withi which to save the resulting plot
    :param y_label: str, y label (value being plotted)
    :param x_label: str, x label (value being changed in each loop)
    """

    fig, ax = plt.subplots(figsize=plot_size())

    for m in range(0, size_array.shape[0]):
        if model_name.shape[0] == 3:
            i = m+1
        else:
            i = m
        ax.plot(x_values, size_array[m, :], label=model_name[m], color=model_colors[i], marker=markers[i],
                linewidth=1, markersize=6)
    handles, labels = ax.get_legend_handles_labels()

    # Y axis
    ax.set_ylabel(y_label)
    if y_label == "Posterior sample size":
        plt.yscale("log")
        plt.grid(b=True, which='minor', color='gainsboro', linestyle='-')
        plt.minorticks_on()
    else:
        if np.max(size_array) / np.min(size_array) > 10:
            plt.yscale("log")
            plt.grid(b=True, which='minor', color='gainsboro', linestyle='-')
            plt.minorticks_on()
        elif np.min(size_array) > 100:
            plt.yscale("log")
            plt.grid(b=True, which='minor', color='gainsboro', linestyle='-')
            plt.minorticks_on()
    if np.min(size_array) < 1 or np.max(size_array) < 100:
        ax.set_ylim(np.min(size_array), np.max(size_array))
    else:
        ax.set_ylim(1, 1000000)

    plt.grid(b=True, which='major', color='lightgrey', linestyle='-')

    # X-axis
    ax.set_xlabel(x_label)
    if np.max(x_values) / np.min(x_values) > 50:
        plt.xscale("log")
        form = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        ax.xaxis.set_major_formatter(form)
        ax.set_xlim(np.min(x_values), 10)

    fig.legend(handles, labels, loc='lower center', ncol=size_array.shape[0])
    plt.subplots_adjust(top=0.95, bottom=0.2, wspace=0.35, hspace=0.25)
    # plt.margins(y=1, tight=True)

    plt.savefig(save_name)

    plt.show(block=False)
    x = 1


def plot_evolving_scores(data_array, x_values, model_names, save_name, uncertainty, x_label):
    """
    Plots how the value of each score changes with increasing number of data points. The values for each score are
    plotted in a different subplot. Each subplot contains curves for each model being analyzed. Each score is plotted
    in a different subplot

    Args:
    -------------------------------------
    :param data_array: 3D np.array 'P' 2D arrays, one for each number of data points analyzed. Each 2D array has size
    [MxS], where M is number of models, and S is the different scores.
    :param x_values: list with the 'x' values (number of data points for which the BI procedure was done)
    :param model_names: list with model names (each curve in each subplot)
    :param save_name: path of file where to save the resulting plot
    :param uncertainty: boolean, if True plots -log(BME) and NNCE, if False plots log(BME) and ELPD
    :param x_label: str, x label (value being changed in each loop)

    Note:
    * Plots BME as -log(BME) to have everything in the same scale
    * User can send only the log(BME) and ELPD scores, to plot only 2 scores, but not any other combination of 2
    """

    cols = 2
    if data_array.shape[2] == 2:
        rows = 1
    else:
        rows = 2
    # Set figure size:
    fig, ax = plt.subplots(figsize=plot_size())

    for s in range(0, data_array.shape[2]):  # Loop through each score
        data = np.transpose(data_array[:, :, s])
        if not uncertainty and (s == 0 or s == 1):
            data = -1 * data

        ax = plt.subplot(rows, cols, s + 1)

        for m in range(0, data.shape[0]):
            y = data[m, :]
            ax.plot(x_values, y, color=model_colors[m], marker=markers[m], linewidth=1, label=model_names[m],
                    markersize=6)
        # X Axis
        if s + 1 <= 2:
            ax.set_xlabel("")  # Remove x label from the upper graphs, for legibility
        else:
            ax.set_xlabel(x_label)

        if np.max(x_values)/np.min(x_values) > 50:
            plt.xscale("log")
            form = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            ax.xaxis.set_major_formatter(form)
            ax.set_xlim(np.min(x_values), math.ceil(np.max(x_values)/10)*10)

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




