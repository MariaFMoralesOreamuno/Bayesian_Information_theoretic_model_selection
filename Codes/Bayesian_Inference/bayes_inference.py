from config import *
from log import *
warnings.filterwarnings("ignore")

class BayesInference:

    """
    Author: María Fernanda Morales Oreamuno

    Class runs BMS analysis, based on Bayes Inference and Information theory

    The equations are based on the following papers:
            Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model
            Selection, Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019.

            Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process
            emulator using information theory. Entropy, X(X), X, 2020,

            Gelman, A., Hwang, J., & Vehtari, A. (2014). Understanding predictive information criteria for Bayesian
            models. Statistics and computing, 24(6), 997-1016.

    Args:
        inst_true_data: instance of 'SyntheticData' class, which contains the measurement values (meas_values),
        covariance matrix (cov_matrix) and point location where to evaluate the models in (point_loc)
        model_num: int with the number of model to evaluate for (any number from 1-5)

    Attributes:
        self.measurement_data: Instance of synthetic data class
        self.model_num: model number (determines which equation/data to evaluate/read)
        self.model_name: model name (string)
        self.prior: prior parameters (MCxN size: MC = num of sets, N=number of parameters)
        self.prior_density: prior sets probability density (MCx1)

        self.output: model outputs (from prior) (MCxT, T=number of measurement points)
        self.likelihood: model likelihood (MCx1)
        self.post_likelihood: posterior likelihood (after rejection sampling)
        self.posterior: posterior values (after rejection sampling)
        self.post_density: posterior densities (after rejection sampling)

        self.BME: BME score (prior)
        self.NNCE: Non-normalized cross entropy or - expected log-predictive density score
        self.RE: Relative entropy score
        self.IE: Information entropy score
        self.CE: Cross entropy from prior to posterior (expected posterior probability density)

    """

    def __init__(self, inst_true_data, model_num):
        self.measurement_data = inst_true_data      # Instance of synthetic data class
        self.model_num = model_num         # model number (determines which equation/data to evaluate/read)
        self.model_name = ""               # model name (string)
        self.prior = np.array([])          # prior parameters (MCxN size: MC = num of sets, N=number of parameters)
        self.prior_density = np.array([])  # prior sets probability density (MCx1)

        self.output = np.array([])           # model outputs (from prior) (MCxT, T=number of measurement points)
        self.likelihood = np.array([])       # model likelihood (MCx1)
        self.post_likelihood = np.array([])  # posterior likelihood (after rejection sampling)
        self.posterior = np.array([])        # posterior values (after rejection sampling)
        self.post_density = np.array([])     # posterior densities (after rejection sampling)

        self.post_output = np.array([])

        self.BME = 0       # BME score (prior)
        self.log_BME = 0   # -log(BME) score
        self.NNCE = 0      # Expected log-predictive density or non-normalized cross entropy (as negative value)
        self.RE = 0        # Relative entropy score
        self.IE = 0        # Information entropy score
        self.CE = 0.0      # Cross entropy from prior to posterior

        logger.debug(f"Generated BayesInference instance for {self.model_name} model")

    def calculate_likelihood(self):
        """
        Function calculates likelihood between measured data (from Syn_Data) and the model output using the stats module
        equations.

        Notes:
        * Generates likelihood array with size [MCx1].
        * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
        errors.

        """
        try:
            likelihood = stats.multivariate_normal.pdf(self.output, cov=self.measurement_data.cov_matrix,
                                                       mean=self.measurement_data.meas_values[0]) # ###########
        except ValueError as e:
            logger.exception(e)
        else:
            self.likelihood = likelihood

        # lh2 = np.full(self.output.shape[0], 0.0)
        # for i in range(0, self.output.shape[0]):
        #     det = np.linalg.det(self.Syn_Data.cov_matrix)  # Calculates det of the covariance matrix
        #     inv = np.linalg.inv(self.Syn_Data.cov_matrix)  # inverse of covariance matrix
        #     diff = self.Syn_Data.meas_data - self.output[i, :]  # Gets the difference between measured and modeled value
        #     term1 = 1 / np.sqrt((math.pow(2 * math.pi, 10)) * det)
        #     term2 = -0.5 * np.dot(np.dot(diff, inv), diff.transpose())
        #     lh2[i] = term1 * np.exp(term2)

    def calculate_likelihood_manual(self):
        """
        Function calculates likelihood between measured data (from Syn_Data) and the model output manually, using numpy
        calculations.

        Notes:
        * Generates likelihood array with size [MCxN], where N is the number of measurement data sets.
        * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
        errors.
        * Method is faster than using stats module.
        """
        # Calculate constants:
        det_R = np.linalg.det(self.measurement_data.cov_matrix)
        invR = np.linalg.inv(self.measurement_data.cov_matrix)
        const_mvn = pow(2 * math.pi, -self.measurement_data.meas_values.shape[1] / 2) * (1 / math.sqrt(det_R)) # ###########

        # vectorize means:
        means_vect = self.measurement_data.meas_values[:, np.newaxis]  # ############

        # Calculate differences and convert to 4D array (and its transpose):
        diff = means_vect - self.output  # Shape: # means
        diff_4d = diff[:, :, np.newaxis]
        transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

        # Calculate values inside the exponent
        inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invR)
        inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
        total_inside_exponent = inside_2.transpose(2, 1, 0)
        total_inside_exponent = np.reshape(total_inside_exponent,
                                           (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))

        likelihood = const_mvn * np.exp(-0.5 * total_inside_exponent)
        if likelihood.shape[1] == 1:
            likelihood = likelihood[:, 0]
        self.likelihood = likelihood

    def rejection_sampling(self):
        """
        Function runs rejection sampling: generating N(MC) uniformly distributed random numbers (RN). If the normalized
        value of the likelihood {likelihood/max(likelihood} is smaller than RN, reject prior sample. The values that
        remain are the posterior.

        Notes:
            *Generates the posterior likelihood, posterior values, and posterior density arrays
            *If max likelihood = 0, then there is no posterior distributions, or the posterior is the same as the
            prior.
        """
        # Generate MC number of random values between 1 and 0 (uniform dist) ---------------------------------------- #
        RN = stats.uniform.rvs(size=self.output.shape[0])    # random numbers
        max_likelihood = np.max(self.likelihood)               # Max likelihood

        # Rejection sampling --------------------------------------------------------------------------------------- #
        if max_likelihood > 0:
            # 1. Get indexes of likelihood values whose normalized values < RN
            post_index = np.array(np.where(self.likelihood / max_likelihood > RN)[0])
            # 2. Get posterior_likelihood:
            self.post_likelihood = np.take(self.likelihood, post_index, axis=0)
            # 3. Get posterior values:
            if self.prior.shape[0] > 0:
                self.posterior = np.take(self.prior, post_index, axis=0)
            # 4. Get posterior density values (and multiply all values in given data set):
            pdf = np.take(self.prior_density, post_index, axis=0)
            self.post_density = np.prod(pdf, axis=1)
            # 5. Get posterior output values:
            self.post_output = np.take(self.output, post_index, axis=0)
        else:
            # All posterior values are equal to prior values:
            self.post_likelihood = self.likelihood
            self.posterior = self.prior
            self.post_density = np.prod(self.prior_density, axis=1)
            self.post_output = self.output

    def calculate_scores(self):
        """
        Calculate all BMS scores: BME (prior), Non-Normalized Cross Entropy (or expected log-predictive density),
        Relative Entropy and Information Entropy.

        Note:
            *If BME = 0, then RE = np.nan, and there are no values for log-predictive density or information entropy
        """
        # BME: ------------------------------------------------------------------------------------------------------ #
        self.BME = np.mean(self.likelihood)
        self.log_BME = -np.log(self.BME)
        # ----------------------------------------------------------------------------------------------------------- #
        if self.BME > 0:
            self.NNCE = -np.mean(np.log(self.post_likelihood))
            self.RE = -np.log(self.BME) - self.NNCE
            self.CE = np.mean(np.log(self.post_density))
            self.IE = -self.RE - self.CE
        # ----------------------------------------------------------------------------------------------------------- #
        else:
            self.NNCE = np.nan
            self.RE = np.nan
            self.IE = np.nan

    @staticmethod
    def generate_mc_runs(MCsize, dist_name, dist_params):
        """
        Generates MC runs for a given input parameter and calculates the probability of each value from the pdf. It
        receives one parameter a time, and thus generates one column at a time.

        Args:
        :param MCsize: number of data sets of a given parameter to generate
        :param dist_name: distribution name, as type string
        :param dist_params: input distribution parameters

        :return: array with prior values (MCsizex1) and array with the probability of each value (MCsizex1)
        """
        if dist_name == "uniform":
            try:
                # Generate prior values
                prior = stats.uniform.rvs(size=MCsize, loc=dist_params[0], scale=dist_params[1] - dist_params[0])
                # get probability of each value in the prior
                pdf = stats.uniform.pdf(prior, loc=dist_params[0], scale=dist_params[1] - dist_params[0])
            except IndexError as e:
                message = "Check input file 'prior_info'. Not enough input parameters for distribution: {}".format(
                    dist_name)
                logger.exception(message)
            except TypeError:
                message = "Wrong data type in distribution {} parameters. Check input.".format(dist_name)
                logger.exception(message)
            else:
                return prior, pdf
        elif dist_name == "norm":
            try:
                prior = stats.norm.rvs(size=MCsize, loc=dist_params[0], scale=dist_params[1])
                pdf = stats.norm.pdf(prior, loc=dist_params[0], scale=dist_params[1])
            except IndexError as e:
                message = "Check input file 'prior_info'. Not enough input parameters for distribution: {}".format(
                    dist_name)
                logger.exception(message)
            except TypeError:
                message = "Wrong data type in distribution {} parameters. Check input.".format(dist_name)
                logger.exception(message)
            else:
                return prior, pdf
        else:
            message = " The {} distribution is not available. Modify code to include it or check input distribution " \
                      "data".format(dist_name)
            logger.exception(message)
            sys.exit()

    @staticmethod
    def generate_prior(MC_size, num_param, parameter_info):
        """
        Function generates MC prior sets, that comprise the prior of the given model

        Args:
        -------------------------------
        :param MC_size: int, number parameter sets to generate
        :param num_param: int, number of parameters
        :param parameter_info: list, with information on the distribution of each parameter set. Each entry in the list
        must contain a list with [distribution_name, parameters]

        :return: np.arrays with the prior and the probability of each parameter (sizes MC_Size x num_param)

        Note: If the num_params < len(parameter_info), then only the first 'num_params' parameters will be used.
        """
        prior = np.full((MC_size, num_param), 0.0)  # array to save prior values for each parameter
        prior_density = np.full((MC_size, num_param), 0.0)  # array to save prob. density for each prior value
        # Loop through each parameter at a time
        for p in range(0, num_param):
            distribution_name = parameter_info[p][0]  # read parameter distribution name
            distribution_params = parameter_info[p][1:]  # read the distribution parameters

            prior[:, p], prior_density[:, p] = BayesInference.generate_mc_runs(MC_size, distribution_name,
                                                                               distribution_params)
        return prior, prior_density

    @staticmethod
    def calculate_model_weight(data_array, order=0):
        """
        Calculates the model weights for one or more scores. Function sums all values column-wise (each column is a
        different model) for each row (each score) and then divides each individual score by said sum to get the model
        weights for each score

        Args:
        ------------------
        :param data_array: Can be either a MxN (M=number of models, N=number of different scores or runs,) to calculate
        for all different runs at once or (Mx1) to calculate for one score at a time
        :param order: determines whether the weighting is row-rise (order=1) or column-wise (order=0)
        -----------------
        :return: array with model weights, either MxS or Mx1 depending on input.

        Note: If order=0, it means that each row corresponds to a different model, and each column is a different 'true
        measurement' data set. In that case, each column sums to 1.
        """
        if data_array.ndim == 1:
            score_array = data_array[np.newaxis, :]

        if order == 0:
            s_sum = np.sum(data_array, axis=0)  # sum all values for each score (each column)
        else:
            s_sum = np.sum(data_array, axis=1)  # sum all values for each score (row-wise)

        # calculate the weight for each model, for each score
        weights = np.divide(data_array, s_sum)

        return weights

    def run_bayes_inference(self):
        """
        Main code that runs the different BMS steps: calculates likelihood, rejection sampling and scores
        """
        logger.debug(f"Run Bayes Inference procedure for {self.model_name} model.")
        # 1. Calculate likelihood
        # self.calculate_likelihood()
        self.calculate_likelihood_manual()

        # 2. Rejection sampling
        self.rejection_sampling()

        # 3. Calculate scores
        self.calculate_scores()

















