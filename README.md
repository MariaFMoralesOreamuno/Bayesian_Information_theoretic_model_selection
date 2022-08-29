# Bayesian and Information-Theoretic Scores for Model Selection and Similarity Analysis
## Introduction
The Python3 algorithms in this repository contain the codes with the application of the methodology for Bayesian model selection and similarity analysis based on Bayesian model evidence (BME) and information theory scores, such as information entropy (IE), expected log-log-predictive density (ELPD) and relative entropy (RE). The methodology was tested on two analytical model setup scenarios and then applied to a groundwater model setup. The measurement data for all model selection scenarios was generated from a realization of one of the competing models, with the goal of providing a controlled setup in which the real data was known. The Bayesian inference methodology was implented using simple Monte Carlo prior-based sampling, with the goal of avoiding additional assumptions. 

## Authors
María Fernanda Morales Oreamuno (maria.morales@iws.uni-stuttgart.de, mf.moraleso92@gmail.com)

Sergey Oladyshkin (sergey.oladyshkin@iws.uni-stuttgart.de)

## References 
Oladyshkin, S., & Nowak, W. (2019). The connection between bayesian inference and information theory for model selection, information gain and experimental design. Entropy, 21(11), 1081. https://doi.org/10.3390/e21111081

Schöniger, A., Illman, W. A., Wöhling, T., & Nowak, W. (2015). Finding the right balance between groundwater model complexity and experimental effort via Bayesian model selection. Journal of Hydrology, 531, 96-110. https://doi.org/10.1016/j.jhydrol.2015.07.047

Schöniger, A., Nowak, W., & Hendricks Franssen, H. J. (2012). Parameter estimation by ensemble Kalman filters with transformed data: Approach and application to hydraulic tomography. Water Resources Research, 48(4). https://doi.org/10.1029/2011WR010462

## Requirements
To install python3 please refer to the intrsuctions in the link [here](https://hydro-informatics.com/python-basics/pyinstall.html).

All of the python3 libraries are initialized in the module *config.py*. 
### Basic libraries
- sys
- os
- pickle
- glob
- time
- logging
- math
- itertools
- warnings

### External libraries
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- tqdm

## Main modules
The repository is composed of four individual python3 files, which contain the name "main" in the file name. These files are *standalone* files, and are run indepenent from each other. Two of the programs correspond to the analytical model scenarios and two to the groundwater model scenarios. 

### main_analytical_models.py 
File corresponds to the Bayesian model selection and similarity analysis for two analytical model scenarios: 
1) 3 competing models, all based on the same equation and different prior probability distributions (U[-5,5], U[-3,3], N[0,1]) or 
2) 5 competing models, all comprised of different equations and with the same prior probability distribution (Uniform[-5,5] for all parameters, for all 5 models)

The analytical model is based on the one presented in Oladyshkin & Nowak (2019). 

The module first generates the synthetic model run, to be taken as the measurement data. It then runs the Bayesian model selection (BMS) analysis first for the total calibration data set size, and then for increasing calibration data set size. Lastly, it then runs the Bayesian model similarity analysis based on Schöniger et al (2015). 

| Input argument | Type | Description |
|----------------|------|-------------|
|`results_path`| *string* | Folder where to save the results |
|`MC_size`| *integer* | Number of Monte Carlo realizations for each competing model| 
|`Nd`| *int* | Number of Monte Carlo realizations for each data-generating model for the model similarity analysis| 
|`scenario_1`| *boolean* | True, if Scenario 1 is to be run, False if Scenario 2 is to be run| 
|`num_measurement_pts`| *int* | Number of data points (observations) to consider (default = 10).|
|`num_param`| *int* | Number of uncertain paramters to consider for all models (default = 10).|
|`measurement_error_val`| *float* | Measurement error for all data points (default = 2).|
|`var_num_mp`| *boolean* | True, to run scenario for the different data set sizes in 'num_dp'| 
|`num_dp`| *list of integers* | list with the calibration data set size for which to run the BMS scenarios| 



### main_trends_analytical_models.py 
Runs the Bayesian model selection analysis for the analytical model mentioned in the previous section, but exclusively to see how the scores vary when a given variable is changed, for example:
a) measurement error or b)number of parameters.


| Input argument | Type | Description |
|----------------|------|-------------|
|`results_path`| *string* | Folder where to save the results |
|`MC_size`| *integer* | Number of Monte Carlo realizations for each competing model| 
|`scenario_1`| *boolean* | True, if Scenario 1 is to be run, False if Scenario 2 is to be run| 
|`num_measurement_pts`| *int* | Number of data points (observations) to consider for the reference model(default = 10).|
|`num_param`| *int* | Number of uncertain paramters to consider for the reference model (default = 10).|
|`measurement_error_val`| *float* | Measurement error for all data points for the reference model(default = 2).|
|`var_error`| *boolean* | True, to run scenario for the different measurement errors in 'err_vals'| 
|`err_vals`| *list of floats* | list with the different measurement errors for which to run the BMS scenarios| 
|`var_num_params`| *boolean* | True, to run scenario for the different number of model parameters in 'num_p'| 
|`num_p`| *list of integers* | list with the different number of parameters for which to run the BMS scenarios| 

### main_groundwater_model.py 
Module runs the Bayesian model selection and model similarity analysis for a set of 5 groundwater models. The following 5 models are considered, they in the ln(K) spatial distribution model and/or the data set size considered: 
- homogeneous transport model: 1 uncertain parameter, 2 output types (hydraulic head and concentration)
- 5-zoned transport model: 5 uncertain parameters, 2 output types (hydraulic head and concentration)
- 5-zoned flow model: 5 uncertain parameters, 1 output types (hydraulic head)
- 9-zoned transport model: 9 uncertain parameters, 2 output types (hydraulic head and concentration)
- geostatistical model: 2500 uncertain parameters, 2 output types (hydraulic head and concentration) 

| Input argument | Type | Description |
|----------------|------|-------------|
|`path_meas`| *string*| Path where .mat file with measurement data is located.|
|`path_Y`| *string* | Path where .mat file with true ln(K) spatial distribution values are located.|
|`path_true_sol`| *string*| Path where true_solution.mat file with the synthetic true measurement data is located. |
|`path_hm`[^1] | *string*| Path where .mat file with data for homogenous model is located. |
|`path_zm`[^1] | *string*| Path where .mat file with data for 5-zoned model is located.|
|`path_zm2`[^1] | *string*| Path where .mat file with data for 9-zoned model is located.|
|`path_gm`[^1] | *string*| Path where .mat file with data for geostatistical model is located.|
|`path_bmj_hm`[^1] | *string*| Path where .mat file with results from the homogenous model (for the model similiarty analysis) is located.|
|`path_bmj_zm`[^1] | *string*| Path where .mat file with  results from the zoned model (for the model similiarty analysis) is located.|
|`path_bmj_gm`[^1] |*string*| Path where .mat file with results from the geostatistical model (for the model similiarty analysis) is located.|
|`results_path` |*string*| Path where results for groundwater models are to be saved. |

[^1]: These files correspond to the prior and output data for each run of the 2D groundwater model. 

## Bayesian and information-theoretic score calculation
The file *bayes_inference.py* contains the classes and corresponding functions to calculate the different Bayesian and information-theoretic scores for Bayesian model selection analysis. 

The file *model_similarity_analysis.py* contains the classes and corresponding functions to calculate the different Bayesian and information-theoretic scores and model confusion matrices (MCM) for Bayesian model similarity analysis. 

The file *measurement_data.py* contains the class to save instances of observed data, which is used by the classes in *bayes_inference.py* and *model_similarity_analysis.py*. 


## Plotting 
The files *plots_fun.py*, *plotsfun_gw.py* and *plot_trends.py* are used to plot the different results for the different stand-alone modules. The figures are generated using Matplotlib (https://matplotlib.org/) with version 3.4.3 and Seaborn (https://seaborn.pydata.org/) with version 0.11.2. 


## Module Outputs
### main_analytical_models.py and main_groundwater_model.py
The program outputs are plots (in pdf format) with the following information: 
- BME values for each competing model
- a) ln(BME), b) ELPD, c) RE and d) IE scores for each competing model in the Bayesian model selection setup
- Relationship between a) ln(BME), ELPD and RE and b) RE, CE and IE for each competing model
- a) prior b) posterior model outputs for each competing model
- prior and posterior parameter distributions for each competing model 
- Model confusion matrix (MCM) for BME weights for the model similarity analysis setup
- MCMC for a) ln(BME), b) ELPD, c) RE and d) IE for the model similarity analysis setup
- Normalized MCMC for a) ln(BME), b) ELPD, c) RE and d) IE for the model similarity analysis setup

For main_analytical_models.py, with var_num_mp=True: 
- Figure with a) evolving BME value for increasing data set size for each competing model b) BME weights for each competing model for increasing data set size
- Evolving a) ln(BME), b) ELPD, c) RE and d) IE scores for each competing model in the Bayesian model selection setup for increasing data set size. 

### main_trends_analytical_models.py and main_trends_groundwater.py
The program outputs are plots (in pdf format) with the following information: 
- Posterior sample size with increasing variable value, for each competing model 
- Evolving a) ln(BME), b) ELPD, c) RE and d) IE scores for each competing model in the Bayesian model selection setup for increasing variable value. 

## Disclaimer
No warranty is expressed or implied regarding the usefulness or completeness of the information and documentation provided. References to commercial products do not imply endorsement by the Authors. The concepts, materials, and methods used in the algorithms and described in the documentation are for informational purposes only. The Authors have made substantial effort to ensure the accuracy of the algorithms and the documentation, but the Authors shall not be held liable, nor his employer or funding sponsors, for calculations and/or decisions made on the basis of application of the scripts and documentation. The information is provided "as is" and anyone who chooses to use the information is responsible for her or his own choices as to what to do with the data. The individual is responsible for the results that follow from their decisions.

This web site contains external links to other external web sites and information provided by third parties. There may be technical inaccuracies, typographical or other errors such as programming bugs or computer viruses contained within the web site or its contents. Users may use the information and links at their own risk. The Authors of this web site exclude all warranties, whether expressed or implied, relating in any way to this website or the use of this web site; and liability (including for negligence) to users in respect of any loss or damage (including special, indirect or consequential loss or damage such as loss of revenue, unavailability of systems or loss of data) arising from or in connection with any use of the information on or accessed through this website for any reason whatsoever (including negligence).
