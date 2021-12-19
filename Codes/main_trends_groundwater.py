"""
Author: María Fernanda Morales Oreamuno
Date created: 17/11/2021

Program runs the groundwater models for different number of model realizations and/or different measurement error values
to determine their effects on posterior sample size, the score values, and the ESS for each measurement type.

User input
--------------------------------------------------
* For error analysis:
error_factor: list of floats, with factors with which to multiply the original error values (the program will run on a
loop through each one)
analyze: list, with the names of the measurement types to include in the analysis
meas_change: list, with the measurement types to change in each loop (can be different than 'analyze', but can't contain
values not included in it.

* For number of realizations:
mc_range: int, number of realizations steps, meaning that the program will run on a loop for every 'mc_range'
realizations up to the max available number, for each error_factor being analyzed
    e.g.: if the total run has 500 000 realizations, and mc_range is 100 000 , the program will be run 5 times, for
    MC numbers = {1e5, 2e5, 3e5, 4e5, 5e5}.

Note: User can choose which measurement types to consider, by changing the list values in variable 'analyze', and the
measurement types' errors to modify, by changing the list values in variable 'meas_change'.
"""
from Bayesian_Inference.bayes_inference import *
from Bayesian_Inference.measurement_data import *
from plots.plot_trends import *


# ------------------------------------------------ FUNCTIONS --------------------------------------------------- #
def calculate_ess(model_list):
    """
    Calculates ESS (expected squared error)

    :param model_list: np.array, with model run instances. Each row is a different model, each column is a different
    error factor

    :return: np.array with ESS for each model, considering all measurement points/types
    """
    ess_array = np.full((model_list.shape[0], model_list.shape[1]), 0.0)

    for m in range(0, model_list.shape[0]):
        for f in range(0, model_list.shape[1]):
            model = model_list[m, f]
            subt = np.subtract(model.post_output, model.measurement_data.meas_values[0, :])
            ess = np.power(subt, 2)
            ess = np.sum(ess, axis=0)
            ess_array[m, f] = np.mean(ess)

    return ess_array


def modify_error(error_array, fact, meas_analyze):
    """
    Function modifies the error values by multiplying the data being analyzed by the corresponding factor

    Args:
    ------------------------
    :param error_array: np array, with error values (size 20x1)
    :param fact: float, value by which to multiply the original error value
    :param meas_analyze: list, of strings, with measurement type being analyzed

    :return: np array, modified error array
    """
    for m in meas_analyze:
        if m == "Y  ":
            error_array[0:5] = error_array[0:5] * fact  # log(K)
        elif m == "h  ":
            error_array[5:10] = error_array[5:10] * fact  # h
        elif m == 'd  ':
            error_array[10:15] = error_array[10:15] * fact  # d
        else:
            error_array[15:20] = (0.02 * fact) + 0.2 * measurement_mat['real_meas'][15:20]

    return error_array


def get_plot_prefix(analyze_list, change_list):
    """
    Function determines the prefix to be used to save all plots

    Args:
    -------------------------------
    :param analyze_list: list, with the data types being analyzed
    :param change_list: list, str with the data types being changed (multiplied by error factor)

    :return: str, prefix
    """
    if len(analyze_list) == 1:
        pref = change_list[0].replace(" ", "")
    else:
        # Analyzing
        pref = "A("
        for i in range(0, len(analyze_list)):
            if i == 0:
                pref = pref + analyze_list[i].replace(" ", "")
            else:
                pref = pref + "_" + analyze_list[i].replace(" ", "")
        if analyze_list != change_list:
            # Changing
            for i in range(0, len(change_list)):
                if i == 0:
                    pref = pref + ")_Ch(" + change_list[i].replace(" ", "")
                else:
                    pref = pref + "_" + change_list[i].replace(" ", "")
        pref = pref + ")"
    return pref


# ------------------------------------------------ USER INPUT --------------------------------------------------- #
# Paths where model data are located (as .mat files)
path_meas = r'' + os.path.abspath(r'../Input/MATLAB_GW/measurements_fest.mat')
path_Y = r'' + os.path.abspath(r'../Input/MATLAB_GW/Y_true_fest.mat')

# BMS data --- -----------------------------------------------------------------------------------------
path_hm = r'' + os.path.abspath(r'../Input/Python_GW/1e6/homogenous_model_final.mat')
path_zm = r'' + os.path.abspath(r'../Input/Python_GW/1e6/5_zoned_model_final.mat')
path_zm2 = r'' + os.path.abspath(r'../Input/Python_GW/1e6/9_zoned_model_final.mat')
path_gm = r'' + os.path.abspath(r'../Input/Python_GW/1e6/geostatistical_model_final.mat')

# Path where to save results -------------------------------------------------------------------
results_path = r'' + os.path.abspath(r'../Results/GW/Trends')

error_factor = [1, 2, 3]
mc_range = 100_000

# Assign type of measurement data ------------------------------------------------------------------------ #
transport_3 = ['Y  ', 'h  ', 'c0 ']
transport_2 = ['h  ', 'c0 ']
flow_2 = ['Y  ', 'h  ']
flow_1 = ['h  ']

# Change the following variables, if needed:
analyze = transport_2    # Which of measurement type lists to consider
meas_change = ['h  ', 'c0 ']    # Measurement type to change in each run


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------- Create Folders ------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

if not os.path.exists(results_path):
    logger.info(f"Creating folder: {results_path}")
    os.makedirs(results_path)


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------- READ MODEL DATA--------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
measurement_mat = loadmat(path_meas)

hm_mat = loadmat(path_hm)
zm_mat = loadmat(path_zm)
gm_mat = loadmat(path_gm)
zm2_mat = loadmat(path_zm2)

# BMS
hm_mat['meas_type'] = analyze
zm_mat['meas_type'] = analyze
gm_mat['meas_type'] = analyze
zm2_mat['meas_type'] = analyze

# Assign name: ----------------------------------------------------- #
hm_mat['name'] = 'hm'
zm_mat['name'] = 'zm\_5'
gm_mat['name'] = 'gm'
zm2_mat['name'] = 'zm\_9'

# models to analyze ------------------------------------------------ #
# mat_list = [hm_mat, zm_mat, zm2_mat, gm_mat]
mat_list = [zm_mat, zm2_mat, gm_mat]

# ----------------------------------- Get constants from MATLAB run ----------------------------------------------- #
num_models = len(mat_list)
MC_size = hm_mat['n_reali'][0, 0]
n_points = measurement_mat['nmeas'][0, 0]/4

if len(analyze) == 1:
    meas_change = analyze

prefix = get_plot_prefix(analyze, meas_change)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------- READ MEASUREMENT DATA------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# Read error  ------------------------------------------------------------------------------------------ #
err = np.sqrt(measurement_mat['merror'])

# Generate class instance --------------------------------------------------------------- #
OM = MeasurementData(measurement_mat['meas_loc'], measurement_mat['measval'], err)
# Set attributes:
OM.meas_type = measurement_mat['meastype']
OM.generate_cov_matrix()
# ----------------------------------------------------------------------------------------------------------- #

# Initiate result arrays
error_vals = np.full(len(error_factor), 0.0)                           # Save error factor or value

model_name = np.full(num_models, "", dtype=object)                     # Save model name
model_scores_plot = np.full((len(error_factor), num_models, 4), 0.0)   # (each row: score, each column: model)
bme_values = np.full((num_models, len(error_factor)), 0.0)             # Array to save original BME values
ce_values = np.full((num_models, len(error_factor)), 0.0)              # Array to save cross entropy

model_runs = np.empty((num_models, len(error_factor)), dtype=object)   # Save BI instances

sample_size = np.full((num_models, len(error_factor)), 0)              # save number of posterior sample size

# Loop through each error factor
for f, factor in enumerate(error_factor):
    if len(error_factor) > 1 and f == 0:
        logger.info("Running groundwater model scenario for increasing measurement error values")
    # Modify original errors
    err = np.sqrt(measurement_mat['merror'])
    if len(analyze) == 1:  # Multiply all errors by factor
        err = modify_error(err, factor, analyze)
    else:
        err = modify_error(err, factor, meas_change)

    OM = MeasurementData(measurement_mat['meas_loc'], measurement_mat['measval'], err)
    OM.meas_type = measurement_mat['meastype']

    # Prepare for all runs with MC
    n_runs = np.arange(1, int(MC_size / mc_range)+1)

    size_post_n = np.full((num_models, n_runs.shape[0]), 0.0)
    scores_n = np.full((n_runs.shape[0], num_models, 4), 0.0)
    mc_values = np.full(n_runs.shape[0], 0)
    # Loop through each number of MC sets
    for n in range(0, n_runs.shape[0]):
        n_mc = mc_range * n_runs[n]
        mc_values[n] = n_mc
        if n_runs.shape[0] > 1 and n == 0:
            logger.info(f"Running groundwater model scenario for increasing MC realizations for an error "
                        f"factor of {factor}")
        # Run BMS
        for modelID, mat in enumerate(mat_list):
            # Determine if measurement values need to be filtered and generate BI instance
            if len(mat['meas_type']) != len(np.unique(OM.meas_type)):
                f_loc, f_vals, f_errors = OM.filter_by_type(mat['meas_type'])
                f_OM = MeasurementData(f_loc, f_vals, f_errors)
                f_OM.generate_cov_matrix()
                f_OM.generate_meas_type(mat['meas_type'], n_points)

                BI = BayesInference(f_OM, modelID + 1)
            else:
                BI = BayesInference(OM, modelID + 1)

            # Assign attributes
            BI.model_name = mat['name']
            try:
                BI.prior = mat['prior']
            except KeyError:
                BI.prior = np.array([])
            BI.prior_density = mat['prior_density']

            # Assign output
            if len(mat['meas_type']) != len(np.unique(OM.meas_type)):
                BI.output = MeasurementData.filter_by_input(mat['output'], OM.meas_type, mat['meas_type'])
                BI.output = BI.output[0:n_mc]
            else:
                BI.output = mat['output']
                BI.output = BI.output[0:n_mc]

            # Run bayes inference
            BI.run_bayes_inference()

            # Save results in arrays
            if n == n_runs.shape[0]-1:  # Save results for total number of MC runs
                error_vals[f] = BI.measurement_data.error[0, 0]

                model_name[modelID] = BI.model_name

                model_scores_plot[f, modelID, :] = np.array([BI.log_BME, BI.NNCE, BI.RE, BI.IE])
                bme_values[modelID, f] = BI.BME
                ce_values[modelID, f] = BI.CE

                model_runs[modelID, f] = BI
                sample_size[modelID, f] = BI.post_likelihood.shape[0]

            # Save always:
            size_post_n[modelID, n] = BI.post_likelihood.shape[0]
            scores_n[n, modelID, :] = np.array([BI.log_BME, BI.NNCE, BI.RE, BI.IE])

    # Plot for current error, for increasing number of MC
    if n_runs.shape[0] > 1:
        plot_name = os.path.join(results_path, f"{prefix}_evolving_scores_MC_error_{factor}.pdf")
        plot_evolving_scores(scores_n, mc_values, model_name, plot_name, uncertainty=False, x_label="Number MC")

        plot_name = os.path.join(results_path, f"{prefix}_evolving_samplesize_MC_error_{factor}.pdf")
        plot_changing_variable(size_post_n, model_name, mc_values, plot_name, "Posterior sample size", "Number MC")


# Prepare for plots: ------------------------------------------------------------------------------------------------
logger.info("Plotting groundwater model setup BMS results for increasing measurement error values, for the total "
            "MC realizations")
if len(analyze) == 1:
    x_label = "Measurement error value"
    x_send = error_vals
else:
    if len(meas_change) == 1:
        x_label = "Measurement error value"
        x_send = error_vals
    else:
        x_label = "Measurement error factor"
        x_send = error_factor

# Print ESS
ess_vals = calculate_ess(model_runs)
plot_name = os.path.join(results_path, f'{prefix}_ESS_errors.pdf')
plot_changing_variable(ess_vals, model_name, x_send, plot_name, "ESS", x_label)

# Print sample size
plot_name = os.path.join(results_path, f'{prefix}_SampleSize_ERRORS.pdf')
plot_changing_variable(sample_size, model_name, x_send, plot_name, "Posterior sample size", x_label)

# Plot evolving scores
scores_no_gm = model_scores_plot[:, :, :]
scores_no_gm[:, -1, -1] = 0.0  # Ignore geostatistical model IE
plot_name = os.path.join(results_path, f'{prefix}_Evolving_scores_ERRORS.pdf')
plot_evolving_scores(scores_no_gm, x_send, model_name, plot_name, uncertainty=False, x_label=x_label)

# Print outputs
for p, type in enumerate(model_runs[0, 0].measurement_data.meas_type):
    plot_name = os.path.join(results_path, f'{prefix}_Output_{type.replace(" ", "")}_dp{p+1}.pdf')
    plot_outputs_gw_trends(model_runs, plot_name, type, p)

stop = 1