"""
Author: María Fernanda Morales Oreamuno

Date created: 24/08/2021
Last modified: 12/12/2021

Program runs the Bayesian model selection and bayesian model similarity analysis for a set of 2D groundwater models.

Based on the following papers:
    Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model
        Selection, Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019.
    Schöniger, A., Illman, W. A., Wöhling, T., & Nowak, W. (2015). Finding the right balance between groundwater model
        complexity and experimental effort via Bayesian model selection. Journal of Hydrology, 531, 96-110.

---------------------------------------------------------------------------------------------------------------------
Generation of GW model results:

Based on: Schöniger, A., Nowak, W., & Hendricks Franssen, H.-J. (2012). Parameter estimation by ensemble Kalman filters
with transformed data: Approach and application to hydraulic tomography. Water Resources Research, 48 (4), -.
https://doi.org/10.1029/2011WR01046

A MATLAB program generates synthetic 'true' measurement data and results for 4 different groundwater models, using a
50x50 m 2D grid:
* Homogenous model: has only one normally distributed uncertain parameter (logK), which is assigned to all cells in the
 model grid.
* 5-Zoned model: has 5 normally distributed uncertain parameters (logK), each pertaining to a different conductivity
zone. The zone classification was done based on the synthetically generated model run.
* 9-Zoned model: has 9 normally distributed uncertain parameters (logK), each pertaining to a different conductivity
zone. The zone classification was done based on the synthetically generated model run.
* Geostatistical model: has 2500 uncertain parameters, which follow a multivariate Gaussian distribution and an
exponential covariance function.
* 'true model': generated using a random realization from the geostatistical model, with noise added to the data.

The MATLAB program calculates head and concentration data in (N) data points, for a total of Nx2 measurement points

 ---------------------------------------------------------------------------------------------------------------------
User input:

* path_meas: string, path where .mat file with measurement data
* path_Y: string, path where .mat file with true Y (log(K)) values are located
* path_true_sol: string, path where true_solution.mat file with the synthetic true measurement data is located
* path_hm: string, path where .mat file with data for homogenous model is located
* path_zm: string, path where .mat file with data for 5-zoned model is located
* path_zm2: string, path where .mat file with data for 9-zoned model is located
* path_gm: string, path where .mat file with data for geostatistical model is located

*path_bmj_hm: path where .mat file with synthetically generated 'true' data from the homogenous model is located
*path_bmj_zm: path where .mat file with synthetically generated 'true' data from the zoned model is located
*path_bmj_gm: path where .mat file with synthetically generated 'true' data from the geostatistical model is located

* results_path: string, path where results for groundwater models are to be saved

----------------------------------------------------------------------------------------------------------------------
Data contained in each .mat file:

* measurements.mat: Contains the measurement data for logK, head, drawdown and concentration, including:
    - nmeas: int, number of measurement data points (DP*4), where DP are the location of measurement points, and there
    are 4 different data types obtained from GW model
    - meastype:np.array with strings indicating the type of measurement data, including 'Y  ' (logK), 'y  ' (head),
    'd  ' (drawdown) and 'c0 ' (concentration)
    - measval: np.array with size [nmeas x 1]
    - meas_loc: np.array with size [nmeas x 3], with [y, x, z] location of each measurement data points
    - merror: np.array with size [nmeasx1] with measurement variance for each data point (square root is the measurement
    error)
    - R: covariance matrix of measurement data

* Y_true.mat: contains log(K) spatial distribution for true synthetic model run
    - Y_true: np.array with log(K) value for each cell in model grid

* true_solution.mat: contains the measurement data, for each cell in the model grid, for each measurement type.
    - h_true: np.array, with size 2601 (for a grid of 51x51) with hydraulic head values
    - m0_true: np.array, with size 2601 (for a grid of 51x51) with concentration values
    - Y_true: np.array, with size 2601 (for a grid of 51x51) with log(K) values

* homogenous_model.mat: Contains data for homogenous model
    - n_reali: int, number of MC realization
    - prior: np.array with uncertain parameters of size [MC x 1]
    - prior_density: np.array with probability of each uncertain parameter of size [MC x 1]
    - output: np.array with model outputs for each MC realization, size [MC, nmeas]

* zoned_model.mat: Contains data for zoned model
    - prior: np.array with uncertain parameters of size [MC x 5]
    - prior_density: np.array with probability of each uncertain parameter of size [MC x 5]
    - output: np.array with model outputs for each MC realization, size [MC, nmeas]
    - zone_grid: np.array with size [50,50], with the zone number of each cell in the model grid.

* geostatistical_model.mat: contains data for geostatistical model
    - prior: np.array with uncertain parameters of size [MC x 2500]  **Optional due to large size
    - prior_density: np.array with probability of each parameter set, with size [MC x 1]
    - output: np.array with model outputs for each MC realization, size [MC, nmeas]
"""

from Bayesian_Inference.bayes_inference import *
from Bayesian_Inference.model_similarity_analysis import *
from plots.plots_fun_gw import *
from plots.results_fun import *

# ------------------------------------------------------------------------------------------------------------------ #
# ---------------------------------------------- INPUT ------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# MATLAB RESULTS ***********************************************************************************************
# Paths where model data are located (as .mat files)
path_meas = r'' + os.path.abspath(r'../Input/MATLAB_GW/measurements_fest.mat')
path_Y = r'' + os.path.abspath(r'../Input/MATLAB_GW/Y_true_fest.mat')
path_true_sol = r'' + os.path.abspath(r'../Input/MATLAB_GW/true_solution_fest.mat')

# Model runs 1e6 -----------------------------------------------------------------------------------------------
path_hm = r'' + os.path.abspath(r'../Input/Python_GW/1e6/homogenous_model_final.mat')
path_zm = r'' + os.path.abspath(r'../Input/Python_GW/1e6/5_zoned_model_final.mat')
path_zm2 = r'' + os.path.abspath(r'../Input/Python_GW/1e6/9_zoned_model_final.mat')
path_gm = r'' + os.path.abspath(r'../Input/Python_GW/1e6/geostatistical_model_final.mat')

# Path for similarity analysis data: ---------------------------------------------------------------------------
path_bmj_hm = r'' + os.path.abspath(r'../Input/Python_GW/1e3/homogenous_model.mat')
path_bmj_zm = r'' + os.path.abspath(r'../Input/Python_GW/1e3/5_zoned_model.mat')
path_bmj_zm2 = r'' + os.path.abspath(r'../Input/Python_GW/1e3/9_zoned_model.mat')
path_bmj_gm = r'' + os.path.abspath(r'../Input/Python_GW/1e3/geostatistical_model.mat')

# Path where to save results -----------------------------------------------------------------------------------
results_path = r'' + os.path.abspath(r'..\Results\GW')


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------- Create Folders ------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

if not os.path.exists(results_path):
    logger.info(f"Creating folder: {results_path}")
    os.makedirs(results_path)

if not os.path.exists(os.path.join(results_path, "runs")):
    logger.info(f"Creating folder: {os.path.join(results_path, 'runs')}")
    os.makedirs(os.path.join(results_path, "runs"))


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------- READ MEASUREMENT DATA------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
measurement_mat = loadmat(path_meas)

# Read error  ------------------------------------------------------------------------------------------ #
err = np.sqrt(measurement_mat['merror'])
# Increase error  ------------------------------------------- #
err[0:5] = err[0:5]*3     # log(K)
err[5:10] = err[5:10]*3   # h
err[15:20] = (0.02*3) + 0.2*measurement_mat['real_meas'][15:20]    # Concentration

# Generate class instance ----------------------------------------------------------------------------- #
OM = MeasurementData(measurement_mat['meas_loc'], measurement_mat['measval'], err)
# Set attributes:
OM.meas_type = measurement_mat['meastype']
OM.generate_cov_matrix()
# ----------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------- READ MODEL DATA--------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
# BMS
hm_mat = loadmat(path_hm)
zm_mat = loadmat(path_zm)
zmf_mat = loadmat(path_zm)
zm9_mat = loadmat(path_zm2)
gm_mat = loadmat(path_gm)

# BMJ
hm_bmj_mat = loadmat(path_bmj_hm)
zm_bmj_mat = loadmat(path_bmj_zm)
zmf_bmj_mat = loadmat(path_bmj_zm)
zm9_bmj_mat = loadmat(path_bmj_zm2)
gm_bmj_mat = loadmat(path_bmj_gm)

# Assign type of measurement data ----------------------------------- #
total_meas_type = ['Y  ', 'h  ', 'd  ', 'c0 ']
transport_meas_type = ['h  ', 'c0 ']
flow_meas_type = ['h  ']

# BMS
hm_mat['meas_type'] = transport_meas_type
zm_mat['meas_type'] = transport_meas_type
zmf_mat['meas_type'] = flow_meas_type
zm9_mat['meas_type'] = transport_meas_type
gm_mat['meas_type'] = transport_meas_type

# BMJ
hm_bmj_mat['meas_type'] = transport_meas_type
zm_bmj_mat['meas_type'] = transport_meas_type
zmf_bmj_mat['meas_type'] = flow_meas_type
zm9_bmj_mat['meas_type'] = transport_meas_type
gm_bmj_mat['meas_type'] = transport_meas_type

# Assign name: ----------------------------------------------------- #
hm_mat['name'] = 'homogeneous'
zm_mat['name'] = '5\_zoned'
zmf_mat['name'] = "5\_zoned\_flow"
zm9_mat['name'] = '9\_zoned'
gm_mat['name'] = 'geostatistical'

hm_mat['reduced_name'] = '$hm$'
zm_mat['reduced_name'] = '$zm_5$'
zmf_mat['reduced_name'] = "$zm_5\_f$"
zm9_mat['reduced_name'] = "$zm_9$"
gm_mat['reduced_name'] = '$gm$'

# models to analyze ------------------------------------------------ #
mat_list = [hm_mat, zm_mat, zmf_mat, zm9_mat, gm_mat]
bmj_mat_list = [hm_bmj_mat, zm_bmj_mat, zmf_bmj_mat, zm9_bmj_mat, gm_bmj_mat]

# ----------------------------------- Get constants from MATLAB run ----------------------------------------------- #
num_models = len(mat_list)
MC_size = hm_mat['n_reali'][0, 0]
n_points = measurement_mat['nmeas'][0, 0]/4
Nd = hm_bmj_mat['n_reali'][0, 0]


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------- PLOT CONSTANTS --------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
logger.info("Plotting synthetic true run and zone classification for zoned models")
plot_synthetic_gw(path_Y, path_true_sol, measurement_mat['meas_loc'], n_points, results_path)

plot_name = os.path.join(results_path, "Zones.pdf")
plot_diff_zones(path_Y, [zm_mat['zone_grid'], zm9_mat['zone_grid']], measurement_mat['meas_loc'], plot_name)

# # Plot individual zone classification grid:
# plot_name = os.path.join(results_path, 'zoned_model_5.pdf')
# plot_zones(path_Y, zm_mat['zone_grid'], measurement_mat['meas_loc'], plot_name)
#
# plot_name = os.path.join(results_path, 'zoned_model_9.pdf')
# plot_zones(path_Y, zm9_mat['zone_grid'], measurement_mat['meas_loc'], plot_name)


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------- Model Selection--------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

#  Initialize results arrays:
model_name = np.full(num_models, "", dtype=object)  # array to save model name
reduced_model_name = np.full(num_models, "", dtype=object)
model_scores_plot = np.full((num_models, 4), 0.0)   # array to save scores to plot (each row: score, each column: model)
bme_values = np.full((num_models, 1), 0.0)          # Array to save original BME values
ce_values = np.full((num_models, 1), 0.0)           # array to save expected posterior density (for plots)

model_runs = []                                     # list to save each model run instance
logger.info("Running Bayesian model selection analysis (BMS) on groundwater models")
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
        logger.warning(f' {BI.model_name} model does not contain prior parameter data set.')
        BI.prior = np.array([])

    try:
        BI.prior_density = mat['prior_density']
    except KeyError:
        logger.error(f' {BI.model_name} model does not contain prior density. Check input data')
        sys.exit()

    # Assign output
    if len(mat['meas_type']) != len(np.unique(OM.meas_type)):
        BI.output = MeasurementData.filter_by_input(mat['output'], OM.meas_type, mat['meas_type'])
    else:
        BI.output = mat['output']

    # Run bayes inference
    BI.run_bayes_inference()

    # print(f'Posterior likelihood size of {BI.model_name} model is: {BI.post_likelihood.shape}')

    # Save results in arrays
    model_name[modelID] = BI.model_name
    reduced_model_name[modelID] = mat['reduced_name']

    model_scores_plot[modelID, :] = np.array([BI.log_BME, BI.NNCE, BI.RE, BI.IE])
    bme_values[modelID, 0] = BI.BME
    ce_values[modelID, 0] = BI.CE

    model_runs.append(BI)


# Save results to .txt files ------------------------------------------------------------------------------------- #
txt_name = os.path.join(results_path, 'runs', 'BMS_GWM.txt')
save_results_txt(model_scores_plot, model_name, txt_name)

# Calculate BME Weights ------------------------------------------------------------------------------------------ #
bme_transport = np.delete(bme_values, 2, axis=0)
bme_weights = BayesInference.calculate_model_weight(bme_transport)
plot_name = os.path.join(results_path, "BME_weights.pdf")
plot_bme_weights_gw(bme_weights, np.array(['']), np.delete(model_name, 2), plot_name)


# ------------------------------------------ Plots --------------------------------------------------------------- #
logger.info("Plotting BMS results for groundwater models")
# 1. Plot all scores in same plot:
plot_name = os.path.join(results_path, 'BMS_scores_GWM.pdf')
if num_models > 3:
    plot_scores_bar_gw(model_scores_plot, reduced_model_name, plot_name, False)
else:
    plot_scores_bar_gw(model_scores_plot, model_name, plot_name)

# ----------------------------------------------------------------------------- #
# 1.1 Remove geostatistical model IE data from results and plot                 #
plot_name = os.path.join(results_path, "BMS_scores_GWM_4M.pdf")                 #
model_scores_mod = np.copy(model_scores_plot)                                   #
model_scores_mod[-1, 3] = 0.0001                                                #
plot_scores_bar_gw(model_scores_mod, reduced_model_name, plot_name, False)      #
# ----------------------------------------------------------------------------- #

# 1.2 Plot individual scores
plot_name = os.path.join(results_path, "BMS_score_GWM_")
plot_individual_scores(model_scores_mod, reduced_model_name, plot_name, uncertainty=False)

# 2. Plot score visualization
# plot_name = os.path.join(results_path, 'BMS_Score_Calculation_GWM.pdf')
# plot_scores_calculations(model_scores_plot, ce_values, model_name, plot_name)
plot_name = os.path.join(results_path, "BMS_stacked_score_relationship_GW.pdf")
plot_stacked_score_calculation_gw(model_scores_plot, ce_values, model_name, plot_name)

# ----------------------------------------------------------------------------------- #
# 2.1 Remove geostatistical model IE data from results and plot                       #
plot_name = os.path.join(results_path, "BMS_stacked_score_relationship_GW_4M.pdf")    #

ce_mod = np.copy(ce_values)                                                           #
ce_mod[-1, 0] = 0.0001                                                                #
plot_stacked_score_calculation_gw(model_scores_mod, ce_mod, model_name, plot_name)    #
# ----------------------------------------------------------------------------------- #

# 3. Plot prior and posterior parameter distributions for Model Selection (MS)
plot_name = os.path.join(results_path, "Prior_Post_GWM.pdf")
plot_prior_post_gw(model_runs[0:-1], plot_name, share=False)

# 4. Plot outputs
n = 0
for i in range(0, int(len(transport_meas_type)*n_points)):
    m_type = model_runs[0].measurement_data.meas_type[i].replace(" ", "")
    plot_name = os.path.join(results_path, f'BMS_output_{m_type}_P{n}.pdf')
    plot_outputs_gw(model_runs, plot_name, model_runs[0].measurement_data.meas_type[i], dp=i)

    if (i+1) % n_points == 0:
        n = 0
    else:
        n = n+1

# 5. Plot likelihoods (post and prior)
plot_name = os.path.join(results_path, "GW_likelihoods.pdf")
plot_likelihoods_gw(model_runs, plot_name)


# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------- Bayesian Model Justifiability Analysis------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------ Generate synthetic runs ----------------------------------------------------------- #
logger.info("Generating synthetic model runs for model similarity analysis: Groundwater models")
true_synthetic_models = []

for modelID in range(0, num_models):
    mat = bmj_mat_list[modelID]
    om_total = MeasurementData(OM.loc, mat['output'], OM.error)
    om_total.generate_meas_type(total_meas_type, n_points)

    # filter by data type
    f_loc, f_vals, f_errors = om_total.filter_by_type(mat['meas_type'])
    data_with_noise = BMJ.synthetic_measurement_noise(f_vals, f_errors)

    om_bmj = MeasurementData(f_loc, data_with_noise, f_errors)
    om_bmj.generate_cov_matrix()
    om_bmj.generate_meas_type(mat['meas_type'], n_points)

    # Add to list ----------------------------------------------------
    true_synthetic_models.append(om_bmj)

# # Plot synthetic runs and MC runs output space
# n = 0
# for i in range(0, int(len(transport_meas_type)*n_points)):
#     m_type = model_runs[0].measurement_data.meas_type[i].replace(" ", "")
#     plot_name = os.path.join(results_path, f'BMS_BMJ_output_{m_type}_P{n}.pdf')
#     plot_outputs_bmj_gw(true_synthetic_models, model_runs, plot_name, model_runs[0].measurement_data.meas_type[i],
#                         dp=i)
#     if (i+1) % n_points == 0:
#         n = 0
#     else:
#         n = n+1

# ---------------------------------- run model similarity analysis ------------------------------------------------- #
logger.info("Running Bayesian model similarity analysis on groundwater models")
bmj_total = BMJ(n_points, Nd, model_runs, true_synthetic_models)
bmj_total.run_bmj()

logger.info("Plotting Bayesian model similarity results for groundwater models")
plot_name = os.path.join(results_path, "BMJ_ConfusionMatrix_GW.pdf")
d_list = [bmj_total.logBME_CM, bmj_total.NNCE_CM, bmj_total.RE_CM, bmj_total.IE_CM]
plot_confusion_matrix_all_scores(d_list, reduced_model_name, plot_name, uncertainty=False, data_type=1,
                                 compat_models=False)

plot_name = os.path.join(results_path, "BMJ_NormConfusionMatrix_GW.pdf")
d_n_list = [bmj_total.BME_norm, bmj_total.NNCE_norm, bmj_total.RE_norm, bmj_total.IE_norm]
plot_confusion_matrix_all_scores(d_n_list, reduced_model_name, plot_name, uncertainty=False, data_type=3)

plot_name = os.path.join(results_path, "BMJ_BMAConfusionMatrix_GW.pdf")
d_list_bme = [bmj_total.BME_CM]
plot_confusion_matrix(d_list_bme, reduced_model_name, 'BME', plot_name, compat_models=False)

for i in range(0, len(d_list)):
    save_name = os.path.join(results_path, 'runs', f"BMJ_{score_list[i]}.txt")
    save_results_txt(d_list[i], model_name, save_name, col_names=model_name)

    save_name = os.path.join(results_path, 'runs', f"BMJ_normal_{score_list[i]}.txt")
    save_results_txt(d_n_list[i], model_name, save_name, col_names=model_name)

save_name = os.path.join(results_path, 'runs', f"BMJ_BME.txt")
save_results_txt(bmj_total.BME_CM, model_name, save_name, col_names=model_name)

logger.info(f"Finished model selection and similarity analysis for groundwater models.")
