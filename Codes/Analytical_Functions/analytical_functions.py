"""
@author: María Fernanda Morales Oreamuno

--- Analytical Non-Gaussian Function and modifications to generate alternative models ---------------------------------
 
The current Code is based on the following manuscripts: 
    Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process emulator using
        information theory. Entropy, X(X), X, 2020,
    Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model Selection, 
        Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019,

For function details and reference information, see:
https://doi.org/10.3390/e21111081
"""

from config import *
from log import *


class AnalyticalFunction:
    """
    The 'AnalyticalFunction' class calculates model outputs for any of the different model possibilities.

    Each model is f(w, t), where 'w' is a set of at least 2 parameters and 't' is time.

    Args:
        model_num: int with number of model to evaluate for (1-5)
        t: np.array with 't' values. Correspond to the time instances in which to obtain the model outputs.
        params: np.array of MCxP size, where MC are the number of MC runs to evaluate the model for (each row) and
        P are the number of parameters for the given model.

    Attributes:
        model_num: int with number of model to evaluate for
        t_loc: np.array with 't' values
        prior: np.array with parameter sets to evaluate the model for.
        prior_density: np.array with probability of each parameter value (only if necessary)

        output: np.array with size MC x len(t) with the model outputs for each parameter set (each row) and for each
        time value (each column)
    """

    def __init__(self, model_num, t, params):
        self.model_num = model_num  # number of model to evaluate for
        self.t_loc = t              # array with time steps to evaluate for
        self.prior = params        # array where the parameters to evaluate are (min 4)
        self.prior_density = np.array([])  # array where the probbaility of each parameter value is

        self.output = np.array([])  # array with model output

    def evaluate_models(self):
        """
        Function evaluates a set of parameters in a given function for different time (x) values.

        Attributes used:
        self.model_num: The model number to evaluate the parameters in
        self.params: an array with param_sets number of rows (one for each MC simulation) and num_params number of
        columns(one for each paramter to evaluate for)
        self.t_loc: time (or x) values to evaluate the given function in

        :return: np.array with the input parameters evaluated in the given model (function) for each time step. The
        resulting array will have a shape of 'param_sets' rows (one for each MC simulation) and 'len(t)' columns (one
        for each time or x value to evaluate for)
        """
        param_sets, num_parameters = self.prior.shape  # rows = # of MC_size, columns: # of parameters
        if self.model_num == 1:
            term1 = (self.prior[:, 0] ** 2 + self.prior[:, 1] - 1) ** 2
            term2 = self.prior[:, 0] ** 2
            term3 = 0.1 * self.prior[:, 0] * np.exp(self.prior[:, 1])
        elif self.model_num == 2:
            term1 = (self.prior[:, 1] - 1) ** 2
            term2 = self.prior[:, 0] ** 2
            term3 = 0.1 * self.prior[:, 0] * np.exp(self.prior[:, 1])
        elif self.model_num == 3:
            term1 = (self.prior[:, 0] ** 2 - 1) ** 2
            term2 = self.prior[:, 0] ** 2
            term3 = 0.1 * self.prior[:, 0] * np.exp(self.prior[:, 1])
        elif self.model_num == 4:
            term1 = (self.prior[:, 0] ** 2 + self.prior[:, 1] - 1) ** 2
            term2 = 0
            term3 = 0.1 * self.prior[:, 0] * np.exp(self.prior[:, 1])
        elif self.model_num == 5:
            term1 = (self.prior[:, 0] ** 2 + self.prior[:, 1] - 1) ** 2
            term2 = self.prior[:, 0] ** 2
            term3 = 0
        else:
            e_message = "Model {} does not exist. Please verify the model number input".format(self.model_num)
            sys.exit(e_message)

        # Term that all models have in common:
        term5 = 0
        if num_parameters > 2:
            for i in range(2, num_parameters):
                term5 = term5 + np.power(self.prior[:, i], 3) / (i + 1)

        # Sum all non-time-related terms: gives one value per row, and one row for each parameter set
        const_per_set = term1 + term2 + term3 + term5 + 1  # All non-time-related terms

        # Calculate time term: gives one value per row for each time interval
        term4 = np.full((param_sets, len(self.t_loc)), 0.0)
        for i in range(0, param_sets):
            term4[i, :] = -2 * self.prior[i, 0] * np.sqrt(0.5 * self.t_loc)

        self.output = term4 + np.repeat(const_per_set[:, None], len(self.t_loc), axis=1)
        return self.output
