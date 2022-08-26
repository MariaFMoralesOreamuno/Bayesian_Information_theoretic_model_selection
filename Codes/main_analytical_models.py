"""
Author: María Fernanda Morales Oreamuno

Date created: 05/11/2021
Last modified: 12/12/2021

Program runs the methodology for Bayesian and information theoretic scores for model selection and model similarity
analysis for an analytical test case, based on the equation used in:
* Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model
  Selection, Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019.

Runs the methodology for 2 possible model comparison scenarios:
* Scenario 1: 3 competing models, all based on the same equation and different prior probability distributions
which are U[-5,5], U[-3,3], N[0,1].
* Scenario 2: 5 competing models, all comprised of different equations and with the same prior probability distribution
(Uniform[-5,5] for all parameters, for all 5 models)

Additionally, the program gives the option to run the scenario for different number of observations (data set sizes),
to compare/analyze the behaviour of the different scores with increasing data set size.

Based on the following papers:
    * Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model
        Selection, Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019.
    * Schöniger, A., Illman, W. A., Wöhling, T., & Nowak, W. (2015). Finding the right balance between groundwater model
        complexity and experimental effort via Bayesian model selection. Journal of Hydrology, 531, 96-110.

Input:
-----------------
- results path: path were to save the model results (plots, .txt summaries, etc).
    A folder will be created within this folder depending on the scenario being run
- MC_size: number of Monte Carlo simulations for the model selection analysis
- Nd: number of Monte Carlo simulations for the data generating models in the BMJ analysis
- scenario_1: boolean, True if Scenario 1 is to be run, False if Scenario 2 is to be run

- num_measurement_pts: int, number of data points to consider
- num_param: int, number of uncertain parameters for all models (all will have the same number of uncertain parameters)
- measurement_error_val: measurement error value for all measurement points

- var_num_mp: bool, True to run for different number of observations
    * num_dp: list, with the number of observations (data points) to consider in each iteration (loop)
"""
from Analytical_Functions.analytical_functions import *
from Bayesian_Inference.bayes_inference import *
from Bayesian_Inference.model_similarity_analysis import *
from plots.plots_fun import *
from plots.results_fun import *

# ----------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- INPUT DATA ------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------- #

results_path = os.path.abspath(r'..\Results')

MC_size = 1_000_000     # number of MC runs
Nd = 1000             # Number of MC runs for the BMJ analysis, for the 'true' model runs

# Set scenario:
scenario_1 = True

# Synthetic true model data ------------------------------------------------------------------------ #
num_measurement_pts = 10    # Number of data points
num_param = 10              # number of uncertain parameters
measurement_error_val = 2   # measurement error value for all data points

# To run for different data set sizes:
var_num_mp = True
num_dp = [1, 5, 8, 10]  # Array with number of data points to analyze in each loop (for multiple data points)


# ----------------------------------------------------------------------------------------------------------------- #
# ---------------------------------- INITIALIZE CONSTANTS --------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #

# Parameter data ----------------------------------------------------------------------------------- #
rg2 = 3      # limits for parameter distribution 2
var_n = 1    # variance for parameter distribution 3
parameter_info_1 = [['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5],
                    ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5], ['uniform', -5, 5]]

parameter_info_2 = [['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2],
                    ['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2], ['uniform', -rg2, rg2],
                    ['uniform', -rg2, rg2], ['uniform', -rg2, rg2]]

parameter_info_3 = [['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n],
                    ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n], ['norm', 0, var_n]]

# extra (in case one wants to add additional models)
mn = 1       # mean
var_n = 2    # variance
parameter_info_4 = [['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n],
                    ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n], ['norm', mn, var_n],
                    ['norm', mn, var_n], ['norm', mn, var_n]]

parameter_info = [parameter_info_1, parameter_info_2, parameter_info_3]  # , parameter_info_4]
param_names = ['U[-5,5]', 'U[-3,3]', 'N[0,1]']
# --------------------------------------------------------------------------------------------------------------- #

# Model-specific data ------------------------------------------------------------------------------------------- #
if scenario_1:
    suffix = "1"
    num_models = 3
    model_name = np.array(param_names)

else:
    suffix = "2"
    num_models = 5
    model_name = np.array(['1', '2', '3', '4', '5'])

results_path = os.path.join(results_path, f"AM_scenario{suffix}")

if not os.path.exists(results_path):
    logger.info(f"Creating folder: {results_path}")
    os.makedirs(results_path)

if not os.path.exists(os.path.join(results_path, "runs")):
    logger.info(f"Creating folder: {os.path.join(results_path, 'runs')}")
    os.makedirs(os.path.join(results_path, "runs"))
# --------------------------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------------------------------- #
# -------------------------------- Run synthetic true data ------------------------------------------------------ #
# --------------------------------------------------------------------------------------------------------------- #
logger.info("Generating synthetic measurement data for BMS analysis")
measurement_error = np.full(num_measurement_pts, measurement_error_val)        # measurement error
time_steps = np.arange(0, num_measurement_pts, 1) / (num_measurement_pts - 1)  # time values for data points
param_values = np.full((1, num_param), 0)  # parameter values to evaluate 'real' model and obtain measurement values

syn_model_run = AnalyticalFunction(1, time_steps, param_values)  # Analytical function class instance
measurement_values = syn_model_run.evaluate_models()  # Evaluate model

# Generate measurement data instance: ------------------------------------------------
syn_data_set = MeasurementData(time_steps, measurement_values, measurement_error)
syn_data_set.generate_cov_matrix()
# --------------------------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- Generate Model Runs --------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------- #

competing_models = []
for m in range(0, num_models):
    if scenario_1:
        param_num = m
        model_num = 1
    else:
        param_num = 0
        model_num = m+1

    # Generate prior
    if scenario_1 or m == 0:
        MC_prior, MC_prior_density = BayesInference.generate_prior(MC_size, num_param, parameter_info[param_num])

    CM = AnalyticalFunction(model_num=model_num, t=time_steps, params=MC_prior)
    CM.prior_density = MC_prior_density
    output = CM.evaluate_models()

    competing_models.append(CM)
# --------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- Bayesian Model Selection-------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
logger.info(f"Running Bayesian model selection analysis (BMS) for scenario {suffix}.")
#  Initialize results arrays:
model_scores_plot = np.full((num_models, 4), 0.0)   # array to save scores to plot (row: model, column: score)
bme_values = np.full((num_models, 1), 0.0)          # array to save original BME values
ce_values = np.full((num_models, 1), 0.0)           # array to save expected posterior density (for plots)

model_runs = []                                     # list to save each model run instance

# Loop through each model at a time and compare to measurement data
for modelID in tqdm(range(0, num_models), desc="Running BMS", unit="Model"):
    # 1. Generate class instance for BayesInference
    BI = BayesInference(syn_data_set, competing_models[modelID].model_num)

    # 2. Set prior (Determine which parameters the given ModelID uses)
    if scenario_1:
        BI.model_name = model_name[modelID]
    else:
        BI.model_name = "Model " + str(model_name[modelID])
    BI.prior = competing_models[modelID].prior
    BI.prior_density = competing_models[modelID].prior_density
    BI.output = competing_models[modelID].output

    # 4. Run bayes inference
    BI.run_bayes_inference()

    # 5. Save results in arrays
    model_scores_plot[modelID, :] = np.array([BI.log_BME, BI.NNCE, BI.RE, BI.IE])
    bme_values[modelID, 0] = BI.BME
    ce_values[modelID, 0] = BI.CE

    model_runs.append(BI)

# Save BMS results ------------------------------------------------------------------------------------------------ #
pck_name = os.path.join(results_path, "runs", 'BMS_results.pkl')
with open(pck_name, 'wb') as f:
    pickle.dump(model_runs, f)

txt_name = os.path.join(results_path, 'runs', 'BMS_results_summary.txt')
save_results_txt(model_scores_plot, model_name, txt_name)

# Calculate BME Weights ------------------------------------------------------------------------------------------- #
bme_weights = BayesInference.calculate_model_weight(bme_values)

plot_name = os.path.join(results_path, "BMS_weights.pdf")
# plot_bme_weights(bme_weights, np.array(['BMS']), model_name, plot_name)

# ------------------------------------------ Plots ----------------------------------------------------------------- #
logger.info(f"Plotting BMS results for scenario {suffix}.")
# Plot each score:
plot_name = os.path.join(results_path, "BMS_scores.pdf")
plot_scores_bar(model_scores_plot, model_name, plot_name, False)

# Plot BME values only
plot_name = os.path.join(results_path, "BME_values.pdf")
plot_bme(bme_values, model_name, plot_name)

# Plot score visualization
plot_name = os.path.join(results_path, "BMS_score_calculation.pdf")
plot_scores_calculations(model_scores_plot, ce_values, model_name, plot_name)

plot_name = os.path.join(results_path, "BMS_stacked_score_relationship_AM.pdf")
plot_stacked_score_calculation(model_scores_plot, ce_values, model_name, plot_name)

# Plot prior and posterior data
plot_name = os.path.join(results_path, "Model_Outputs.pdf")
plot_outputs(model_runs, plot_name)

save_name = os.path.join(results_path, "Prior_Post.pdf")
plot_prior_post(model_runs, save_name)

# # Plot likelihoods
# plot_likelihoods(model_runs, 'Likelihoods_MS', results_path)
# ------------------------------------------------------------------------------------------------------------------ #


# ----------------------------------------------------------------------------------------------------------------- #
# ------------------------------------ Bayesian Model Selection  -------------------------------------------------- #
# ---------------------------- With different calibration data set sizes ------------------------------------------ #
if var_num_mp:
    logger.info(f"Running Bayesian model selection analysis (BMS) for different calibration data set size"
                f" for scenario {suffix}.")

    original_index = np.arange(0, num_measurement_pts, 1)  # Index values of total number of measurement points

    # Create 3D arrays to save each score array of size (NxM) --> N: number of data point values, M: number of models
    results_mp = np.full((len(num_dp), num_models, 4), 0.0)
    results_bme_mp = np.full((num_models, len(num_dp)), 0.0)
    results_ce_mp = np.full((num_models, len(num_dp)), 0.0)

    results_models = np.empty((len(num_dp), len(model_runs)), dtype=object)

    # Loop for each number of data points to analyze
    for n in range(0, len(num_dp)):
        # Get all possible combinations of the given number of points:
        combos = list(itertools.combinations(original_index, num_dp[n]))

        # Loop through each combination and then through each model
        score_results_c = np.full((len(combos), num_models, 4), 0.0)  # 3D array to save all results for 'n' data values
        bme_results_c = np.full((num_models, len(combos)), 0.0)

        # Loop through each combination
        d = "Running for each {} measurement point combination".format(num_dp[n])
        for c in tqdm(range(0, len(combos)), desc=d, unit='Comb.'):
            combi = combos[c]
            # ------------- Set synthetic measurement data: ------------- #
            # Set values to send to synthetic class instance: filter original data to the indexes in "c"
            f_loc, f_val, f_err = syn_data_set.filter_by_index(combi)
            s_data = MeasurementData(f_loc, f_val, f_err)
            s_data.generate_cov_matrix()

            # Loop through each model: here running through each original model instance run previously
            for k, orig_model in enumerate(model_runs):
                # 1. Get data from previously run models (previously run), filtering data by index
                f_bi = BayesInference(s_data, orig_model.model_num)
                f_bi.model_name = orig_model.model_name

                # 2. Set prior
                f_bi.prior = orig_model.prior  # Get from unique MC sampling
                f_bi.prior_density = orig_model.prior_density  # Get from unique MC sampling

                # 3. Generate model outputs: filter from "model.output" (which was previously run)
                f_bi.output = np.take(orig_model.output, combi, axis=1)

                # 4. Run bayes inference
                f_bi.run_bayes_inference()

                # 5. Save results:
                score_results_c[c, k, :] = np.array([f_bi.log_BME, f_bi.NNCE, f_bi.RE, f_bi.IE])
                bme_results_c[k, c] = f_bi.BME

                if c == 0:
                    results_models[n, k] = f_bi
                    results_ce_mp[k, n] = f_bi.CE

        # After all combinations are run, get average for each column (each possible combination, for each score)
        results_mp[n, :, :] = np.mean(score_results_c, axis=0)
        results_bme_mp[:, n] = np.mean(bme_results_c, axis=1)

    # Save results to files --------------------------- ------------------------------------------------------------ #
    txt_name = os.path.join(results_path, 'runs')
    save_results_txt(results_mp, model_name, txt_name, suffix=num_dp)

    # Set names for plots:
    if scenario_1:
        m_names = model_name
    else:
        m_names = np.full(len(model_name), "M", dtype=np.str)
        m_names = np.char.add(m_names, model_name)

    logger.info(f"Plotting BMS results for increasing calibration data set size for scenario {suffix}.")

    # Get BME weights and plot ------------------------------------------------------------------------------------- #
    bme_weights_mp = BayesInference.calculate_model_weight(results_bme_mp)
    # plot_name = os.path.join(results_path, "\\BME_weights_mp.pdf")
    # plot_bme_weights(bme_weights_mp, np.array(num_dp), model_name, plot_name)

    # Plot evolving values for each score:
    plot_name = os.path.join(results_path, "EvolvingScores_mp.pdf")
    plot_evolving_values(results_mp, num_dp, model_name, plot_name, False)

    plot_name = os.path.join(results_path, "Evolving_BME_mp.pdf")
    plot_evolving_bme(results_bme_mp, bme_weights_mp, num_dp, model_name, plot_name)

    plot_name = os.path.join(results_path, "EvolvingScores_calculations_mp.pdf")
    plot_evolving_calculations(results_mp, results_ce_mp, num_dp, plot_name, m_names, all_models=True)

    if scenario_1:
        results_norm = results_mp[:, 2, :].reshape(len(num_dp), 1, results_mp.shape[2])
        plot_name = os.path.join(results_path, "EvolvingScores_Calculations_Normal_mp.pdf")
        plot_evolving_calculations(results_norm, results_ce_mp[2, :].reshape(1, len(num_dp)), num_dp, plot_name,
                                   np.array([model_name[2]]), all_models=False)

        # Plot evolving prior/posterior values
        plot_name = os.path.join(results_path,"Prior_Post_NormalDist_mp.pdf")
        plot_prior_post(results_models[:, 2], plot_name)  # Model N[0,1]
    else:
        plot_name = os.path.join(results_path, "EvolvingScores_Calculations_M1.pdf")
        plot_evolving_calculations(results_mp, results_ce_mp, num_dp, plot_name, m_names, all_models=False)
        # Plot evolving prior/posterior values
        plot_name = os.path.join(results_path, "Prior_Post_M1_mp.pdf")
        plot_prior_post(results_models[:, 0], plot_name)  # Model 1

    # plot_likelihoods(results_models[:, 0], "Likelihoods_Model1_DP", plot_results)  # Model 1
    # plot_lpd(results_models)
    # plot_lpd(results_models[:, 0, None])
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------- Bayesian Model Justifiability Analysis------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------ Generate synthetic runs ----------------------------------------------------------- #
logger.info(f"Generating synthetic data sets for model similarity analysis for scenario {suffix}.")
true_synthetic_models = []
num_dp = [10]
for modelID in range(0, num_models):
    if scenario_1:
        param_num = modelID
        model_num = 1
    else:
        param_num = 0
        model_num = modelID+1

    # Generate priors:
    MC_Nd, MC_Nd_density = BayesInference.generate_prior(Nd, num_param, parameter_info[param_num])

    # Run analytical equation for modelID
    syn_model_run = AnalyticalFunction(model_num, time_steps, MC_Nd)
    syn_output = syn_model_run.evaluate_models()

    # Add noise to the 'synthetic' measurement values:
    syn_data_w_noise = BMJ.synthetic_measurement_noise(syn_output, measurement_error)

    # Generate synthetic data class instance:
    syn_run = MeasurementData(time_steps, syn_data_w_noise, measurement_error)
    syn_run.generate_cov_matrix()

    # Save model run in list:
    true_synthetic_models.append(syn_run)

# --------------------------------------- Run BMJ  ----------------------------------------------------------------- #
logger.info(f"Running Bayesian model similarity analysis for scenario {suffix}.")
# For only one data point
if len(num_dp) == 1:
    # Generate instance of BMJ for all number of data points.
    bmj_1 = BMJ(num_dp[0], Nd, model_runs, true_synthetic_models)

    # Run BMJ
    bmj_1.run_bmj()

    # Plot
    logger.info(f"Plotting model similarity analysis results for scenario {suffix}.")
    # Only BME:
    plot_name = os.path.join(results_path, "BMJ_BME_weights_AM_mp.pdf")
    plot_confusion_matrix([bmj_1.BME_CM], model_name, "BME", plot_name, use_labels=True)

    # BMJ values:
    plot_name = os.path.join(results_path, "BMJ_ConfusionMatrix_AM_mp.pdf")
    d_list = [bmj_1.logBME_CM, bmj_1.NNCE_CM, bmj_1.RE_CM, bmj_1.IE_CM]  # with -log(BME)
    plot_confusion_matrix_all_scores(d_list, model_name, plot_name, False, data_type=1)

    # BMJ normalization
    plot_name = os.path.join(results_path, "BMJ_Norm_ConfusionMatrix_AM_mp.pdf")
    d_n_list = [bmj_1.BME_norm, bmj_1.NNCE_norm, bmj_1.RE_norm, bmj_1.IE_norm]
    plot_confusion_matrix_all_scores(d_n_list, model_name, plot_name, False, data_type=3)

    # Save results as .txt files:
    save_name = os.path.join(results_path, "runs", "BMJ_BME_weights.txt")
    save_results_txt(bmj_1.BME_CM, model_name, save_name, col_names=model_name)

    for i in range(0, len(d_list)):
        save_name = os.path.join(results_path, "runs", f"BMJ_{score_list[i]}.txt")
        save_results_txt(d_list[i], model_name, save_name, col_names=model_name)

        save_name = os.path.join(results_path, "runs", f"BMJ_normal_{score_list[i]}.txt")
        save_results_txt(d_n_list[i], model_name, save_name, col_names=model_name)

logger.info(f"Finished model selection and similarity analysis for scenario {suffix}.")
