"""
Author: María Fernanda Morales Oreamuno

Created: 06/10/2021
Modified: 08/12/2021

Program runs the methodology for Bayesian and information theoretic scores for model selection and model similarity
analysis for an analytical test case, based on the equation used in:
* Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model
  Selection, Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019.

Program runs the analytical model scenarios from "main_analytical_models.py", either for scenario 1 (multiple priors) or
scenario 2 (multiple equations), to see how the scores vary when a given variable is changed, for example:
a) measurement error or b)number of parameters.


User input:
------------------------------------------------
results_path: str, path where the plots/results are to be saved
MC_size:      int, number of Monte Carlo realizations
-scenario_1:  bool, True to run scenario 1, and False to run scenario 2

-var_error : bool, True to run for different error values
    * err_vals: list of floats/ints, with the measurement error values with which to run the models

-var_num_params: bool, True to run for different number of model parameters
    * num_p: list of int, with number of parameters to run the models for (minimum 2 parameters must be set)

Based on the following papers:
    * Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model
        Selection, Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019.
    * Schöniger, A., Illman, W. A., Wöhling, T., & Nowak, W. (2015). Finding the right balance between groundwater model
        complexity and experimental effort via Bayesian model selection. Journal of Hydrology, 531, 96-110.
"""

from Analytical_Functions.analytical_functions import *
from Bayesian_Inference.measurement_data import *
from Bayesian_Inference.bayes_inference import *

from plots.plot_trends import *


# ----------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- INPUT DATA ------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------- #

results_path = r'' + os.path.abspath(r'..\Results')

MC_size = 1_000_000  # number of MC runs

# Set scenario:
scenario_1 = True

var_error = True
err_vals = [5, 2, 1, 0.1, 0.01]

var_num_params = True
num_p = [2, 5, 8, 10]

# ------------------------------------------------------------------------------------------------------------------ #
# ---------------------------------------- Measurement Data -------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# 1: Generate constant/base measurement data:  ------------------------------------------------------------------- #
num_measurement_pts = 10   # Number of data points
num_param = 10             # number of uncertain parameters
measurement_error = np.full(num_measurement_pts, 2)  # measurement error
time_steps = np.arange(0, num_measurement_pts, 1) / (num_measurement_pts - 1)  # time values for data points
param_values = np.full((1, num_param), 0)  # parameter values to evaluate 'real' model and obtain measurement values


# -------------------------------------------------------------------------------------------------------------- #
# --------------------------------- Input parameter information ---------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# Array with information on parameter distribution [name, parameters]
rg2 = 3      # limits for parameter distribution 2
var_n = 1    # variance for parameter distribution 3
parameter_info_1 = [['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5],
                    ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5]]

parameter_info_2 = [['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2],
                    ['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2],
                    ['uniform', -rg2, rg2], ['uniform', -rg2, rg2]]

parameter_info_3 = [['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n],
                    ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n]]


# extra (in case one wants to add additional models
mn = 1       # mean
var_n = 2    # variance
parameter_info_4 = [['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n],
                    ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n]]

parameter_info = [parameter_info_1, parameter_info_2, parameter_info_3, parameter_info_4]

# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# -------------------- Assign values to change---------------------------------------------------------------------- #
if scenario_1:
    num_models = 3  # CHANGE TO INCLUDE ADDITIONAL MODELS (PRIORS)
    model_name = ['U[-5,5]', 'U[-3,3]', 'N[0,1]', f'N[{mn},{var_n}]']
    suffix = "AM1"

    results_path = os.path.join(results_path, f'AM_scenario1', "Trends")

else:
    num_models = 5
    model_name = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
    suffix = "AM2"

    results_path = os.path.join(results_path, f'AM_scenario1', "Trends")


# Create additional folders ---------------------------------------------------------------------------------------- #
if not os.path.exists(results_path):
    logger.info(f"Creating folder: {results_path}")
    os.makedirs(results_path)

if not os.path.exists(os.path.join(results_path, "runs")):
    logger.info(f"Creating folder: {os.path.join(results_path, 'runs')}")
    os.makedirs(os.path.join(results_path, "runs"))

# ------------------------------------------------------------------------------------------------------------------ #
# Run BMS with varying error value --------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

if var_error:
    model_scores_plot = np.full((len(err_vals), num_models, 4), 0.0)  # array to save scores to plot
    bme_values = np.full((num_models, len(err_vals)), 0.0)  # Array to save original BME values
    ce_values = np.full((num_models, len(err_vals)), 0.0)  # array to save expected posterior density (for plots)
    post_size = np.full((num_models, len(err_vals)), 0)

    logger.info(f"Running BMS for analytical model scenario {suffix} for increasing measurement error values")
    for e in trange(0, len(err_vals), desc=f"Running for each error value"):
        # set error value
        measurement_error_v = np.full(num_measurement_pts, err_vals[e])

        # Run real model run:
        real_model_run = AnalyticalFunction(1, time_steps, param_values)  # Analytical function class instance
        measurement_values = real_model_run.evaluate_models()  # Evaluate model

        # Synthetic data: ----------------------------------------------
        # Generate synthetic (true) data class with input/read measurement data:
        true_data_set = MeasurementData(time_steps, measurement_values, measurement_error_v)
        true_data_set.generate_cov_matrix()

        # Generate model runs:
        competing_models = []
        for m in range(0, num_models):
            if scenario_1:
                param_id = m
                model_id = 1
            else:
                param_id = 1
                model_id = m+1

            MC_prior, MC_prior_density = BayesInference.generate_prior(MC_size, num_param, parameter_info[param_id])

            CM = AnalyticalFunction(model_num=model_id, t=time_steps, params=MC_prior)
            CM.prior_density = MC_prior_density
            op = CM.evaluate_models()

            competing_models.append(CM)

        # Run BMS ------------------------------------------------------------------------------------------------ #

        for modelID in range(0, num_models):
            # 1. Generate class instance for BayesInference
            if scenario_1:
                BI = BayesInference(true_data_set, 1)
            else:
                BI = BayesInference(true_data_set, modelID+1)

            # 2. Set prior
            BI.model_name = model_name[modelID]
            BI.prior = competing_models[modelID].prior
            BI.prior_density = competing_models[modelID].prior_density

            # 3. Generate model outputs
            model_run = AnalyticalFunction(BI.model_num, BI.measurement_data.loc, BI.prior)
            BI.output = model_run.evaluate_models()
            # print("Model run: \n", BI.output)

            # 4. Run bayes inference
            BI.run_bayes_inference()

            # 5. Save results in arrays
            model_scores_plot[e, modelID, :] = np.array([BI.log_BME, BI.NNCE, BI.RE, BI.IE])
            bme_values[modelID, e] = BI.BME
            ce_values[modelID, e] = BI.CE
            post_size[modelID, e] = np.size(BI.post_likelihood)

    logger.info(f'Plotting results for analytical model scenario {suffix} for increasing error values')
    plot_name = os.path.join(results_path, f"PosteriorSize_ERRORS_{suffix}.pdf")
    plot_changing_variable(post_size, np.array(model_name), err_vals, plot_name, 'Posterior sample size',
                           'Measurement error value')

    plot_name = os.path.join(results_path, f"Score_trend_ERRORS_{suffix}.pdf")
    plot_evolving_scores(model_scores_plot, err_vals, model_name, plot_name, True, 'Measurement error value')

    plot_name = os.path.join(results_path, f"Score_trend_ERRORS_BME_ELPD_{suffix}.pdf")
    plot_evolving_scores(model_scores_plot[:, :, 0:2], err_vals, model_name, plot_name, True, 'Measurement error value')

if var_num_params:
    logger.info(f"Running BMS for analytical model scenario {suffix} for increasing number of parameters")
    model_scores_plot_p = np.full((len(num_p), num_models, 4),
                                  0.0)  # array to save scores to plot (each row: score, each column: model)
    bme_values_p = np.full((num_models, len(num_p)), 0.0)  # Array to save original BME values
    ce_values_p = np.full((num_models, len(num_p)), 0.0)  # array to save expected posterior density (for plots)
    post_size_p = np.full((num_models, len(num_p)), 0)

    d = f"Running for each number of parameters"
    for p in trange(0, len(num_p), desc=d):
        # set error value
        num_param_v = num_p[p]

        # Run real model run:
        real_model_run = AnalyticalFunction(1, time_steps, param_values)  # Analytical function class instance
        measurement_values = real_model_run.evaluate_models()  # Evaluate model

        # Synthetic data: ----------------------------------------------
        # Generate synthetic (true) data class with input/read measurement data:
        true_data_set = MeasurementData(time_steps, measurement_values, measurement_error)
        true_data_set.generate_cov_matrix()

        # Generate model runs:
        competing_models = []
        for m in range(0, num_models):
            if scenario_1:
                param_id = m
                model_id = 1
            else:
                param_id = 1
                model_id = m+1

            MC_prior, MC_prior_density = BayesInference.generate_prior(MC_size, num_param_v, parameter_info[param_id])

            CM = AnalyticalFunction(model_num=model_id, t=time_steps, params=MC_prior)
            CM.prior_density = MC_prior_density
            op = CM.evaluate_models()

            competing_models.append(CM)

        # Run BMS ------------------------------------------------------------------------------------------------ #

        for modelID in range(0, num_models):
            # 1. Generate class instance for BayesInference
            if scenario_1:
                BI = BayesInference(true_data_set, 1)
            else:
                BI = BayesInference(true_data_set, modelID+1)

            # 2. Set prior
            BI.model_name = model_name[modelID]
            BI.prior = competing_models[modelID].prior
            BI.prior_density = competing_models[modelID].prior_density

            # 3. Generate model outputs
            model_run = AnalyticalFunction(BI.model_num, BI.measurement_data.loc, BI.prior)
            BI.output = model_run.evaluate_models()
            # print("Model run: \n", BI.output)

            # 4. Run bayes inference
            BI.run_bayes_inference()

            # 5. Save results in arrays
            model_scores_plot_p[p, modelID, :] = np.array([BI.log_BME, BI.NNCE, BI.RE, BI.IE])
            bme_values_p[modelID, p] = BI.BME
            ce_values_p[modelID, p] = BI.CE
            post_size_p[modelID, p] = np.size(BI.post_likelihood)

    logger.info(f'Plotting results for analytical model scenario {suffix} for increasing number of parameters')
    plot_name = os.path.join(results_path, f"Score_trend_PARAMETERS_{suffix}.pdf")
    plot_evolving_scores(model_scores_plot_p, num_p, model_name, plot_name, True, 'Number of parameters')

    x = 1




