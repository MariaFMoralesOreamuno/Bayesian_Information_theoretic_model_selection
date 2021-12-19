# Bayesian and Information-theoretic Scores for Model Selection and Similarity Analysis
## Introduction
The Python3 algorithms in this repository accompany the master thesis "Bayesian and Information-Theoretic Scores for Model Selection and Similarity Analysis" and contain a methodology for Bayesian model selection and similarity analysis based on Bayesian model evidence (BME) and information theory scores, such as information entropy, expected log-log-predictive density and relative entropy. The methodology was tested on two analytical model setup scenarios and a set of groundwater models. The measurement data for all model selection scenarios was generated from a realization of one of the competing models, with the goal of providing a controlled setup in which the real data was known. The Bayesian inference methodology was implented using simple Monte Carlo prior-based sampling, with the goal of avoiding assumptions. 

## Author
María Fernanda Morales Oreamuno

Contact: mf.moraleso92@gmail.com

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
File corresponds to the Bayesian model selection and similarity analysis for the two analytical model scenarios. The module first generates the synthetic model run, to be taken as the measurement data. It then runs the Bayesian model selection (BMS) analysis first for the total calibration data set size, and then for increasing calibration data set size. It then runs the Bayesian model similarity analysis. 

| Input argument | Type | Description |
|----------------|------|-------------|
|`results_path`| *string* | Folder where to save the results |
|`MC_size`| *integer* | number of Monte Carlo realizations for each competing model| 
|`num_dp`| *integer list* | list with the calibration data set size for which to run the BMS scenarios| 
|`scenario_1`| *boolean* | True, if Scenario 1 is to be run, False if Scenario 2 is to be run| 

