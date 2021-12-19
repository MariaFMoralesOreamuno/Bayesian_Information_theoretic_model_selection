from Bayesian_Inference.measurement_data import *

warnings.filterwarnings("ignore")


class BayesInferenceBMJ:
    """
    Class runs BMS analysis, based on Bayes Inference and Information theory, for the model similarity analysis

    The equations are based on the following papers:
        Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model
        Selection, Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019.

        Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process
        emulator using information theory. Entropy, X(X), X, 2020,

        Gelman, A., Hwang, J., & Vehtari, A. (2014). Understanding predictive information criteria for Bayesian
        models. Statistics and computing, 24(6), 997-1016.

    Args:
    inst_syn_data: instance of 'MeasurementData' class, which contains the measurement values (meas_values),
            covariance matrix (cov_matrix) and point location where to evaluate the models in (loc)
    model_num: int with the number of model to evaluate for (any number from 1-5)

    Attributes:
    self.measurement_data: Instance of MeasurementData class
    self.prior: prior parameters (MCxN size: MC = num of sets, N=number of parameters)
    self.prior_density: prior sets probability density (MCx1)

    self.output: model outputs (from prior) (MCxT, T=number of measurement points)
    self.likelihood: model likelihood (MCx1)
    self.post_likelihood: posterior likelihood (after rejection sampling)
    self.post_density: posterior densities (after rejection sampling)

    self.BME: BME score (prior)
    self.NNCE: Non-normalized cross entropy or -expected log-predictive density score
    self.RE: Relative entropy score
    self.IE: Information entropy score
    """

    def __init__(self, inst_syn_data, model_num):
        self.measurement_data = inst_syn_data  # Instance of MeasurementData class
        self.model_name = ""
        self.prior = np.array([])  # prior parameters (MCxN size: MC = num of sets, N=number of parameters)
        self.prior_density = np.array([])  # prior sets probability density (MCx1)

        self.output = np.array([])  # model outputs (from prior) (MCxT, T=number of measurement points)
        self.likelihood = np.array([])  # model likelihood (MCx1)
        self.post_likelihood = np.array([])  # posterior likelihood (after rejection sampling)
        self.post_density = np.array([])  # posterior densities (after rejection sampling)

        self.BME = 0  # BME score (prior)
        self.NNCE = 0  # Expected log-predictive density or non-normalized cross entropy (as negative value)
        self.RE = 0  # Relative entropy score
        self.IE = 0  # Information entropy score

    def calculate_likelihood_multiple_means(self):
        """
        Function calculates the likelihood between a set of synthetical measurement values and model outputs.

        Notes:
        *Measurement value array has a size NdxNp, where Nd is the number of synthetic measurement data set and Np is
        the number of measurement points for the given model.
        * Since there are multiple measurement sets, and in order to avoid loops, the stats.multivariate_normal.pdf
        function is not used.
        *Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
        errors.
        """
        # Calculate constants:
        det_R = np.linalg.det(self.measurement_data.cov_matrix)
        invR = np.linalg.inv(self.measurement_data.cov_matrix)
        const_mvn = pow(2 * math.pi, -self.measurement_data.meas_values.shape[1] / 2) * (1 / math.sqrt(det_R))

        # Loop through each measurement data set (to avoid large matrices)
        likelihood = np.full((self.output.shape[0], self.measurement_data.meas_values.shape[0]), 0.0)
        for j in range(0, self.measurement_data.meas_values.shape[0]):
            diff = np.subtract(self.measurement_data.meas_values[j, :], self.output)[:, np.newaxis]
            diff_transp = diff.transpose(0, 2, 1)

            inside = np.einsum("abc, cc->abc", diff, invR)
            inside = np.einsum("abc, acb->ab", inside, diff_transp)
            lik = const_mvn * np.exp(-0.5 * inside)
            likelihood[:, j] = lik[:, 0]

        return likelihood

    def posterior_density_vectorize(self):
        """
        Function calculates the posterior densities, for multiple measurement data sets.

        :return: np.array with size (1 x Nd), where Nd is the number of monte carlo sets considered as 'true' synthetic
        model sets.

        Note: Assigns a value of np.nan for values that are not in the posterior, and the probability value for each
        posterior parameter value. All the columns in a row contain either np.nan or the pdf value.
        """
        # ind_1 = np.argwhere(~np.isnan(self.post_likelihood[:, 0]))

        # Vectorize posterior likelihoods
        vect_likelihood = self.post_likelihood[np.newaxis, :].transpose(2, 1, 0)
        vect_likelihood = np.divide(vect_likelihood, vect_likelihood)  # to get values of 1 for posterior values

        # Get prior densities (multiply all parameter probabilities for each parameter set or row )
        pdf_prior = np.prod(self.prior_density, axis=1)
        pdf_prior = pdf_prior.reshape(pdf_prior.shape[0], 1)

        # Calculate posterior densities
        post_density = np.einsum('kij, ij->kij', vect_likelihood, pdf_prior)
        post_density = post_density.transpose(2, 1, 0).reshape(post_density.shape[1], post_density.shape[0])

        return post_density

    def rejection_sampling_vectorize(self):
        """
        Function runs rejection sampling for multiple 'synthetic true' measurement data sets:
        Generates N(MC) uniformly distributed random numbers (RN). If the normalized value of the likelihood
        {likelihood/max(likelihood} is smaller than RN, rejects prior sample and assigns a value of np.nan to cell. The
        values that remain (have values) are the posterior.

        Notes:
            * Generates the posterior likelihood and posterior density arrays, which are needed for BMJ analysis
            * If max likelihood = 0, then there is no posterior distributions, or the posterior is the same as the
            prior.
        """
        num_meas_data_sets = self.measurement_data.meas_values.shape[0]

        # Generate MC number of random values between 1 and 0 (uniform dist) ---------------------------------------- #
        RN = stats.uniform.rvs(size=[self.output.shape[0], num_meas_data_sets])  # random numbers
        max_likelihood = np.max(self.likelihood, axis=0)  # Max likelihood

        # Rejection sampling:
        with warnings.catch_warnings():
            self.post_likelihood = np.where(np.divide(self.likelihood, max_likelihood) > RN, self.likelihood, np.nan)
            self.post_density = self.posterior_density_vectorize()

    def calculate_scores(self):
        """
        Calculate all BMS scores: BME (prior), Non-Normalized Cross Entropy (or - expected log-predictive density),
        Relative Entropy and Information Entropy.

        Note:
            *If BME = 0, then RE = np.nan, and there are no values for log-predictive density or information entropy
        """
        # BME: ------------------------------------------------------------------------------------------------------ #
        self.BME = np.mean(self.likelihood, axis=0)

        # Information-theoretic scores: ----------------------------------------------------------------------------- #
        self.NNCE = -np.nanmean(np.log(self.post_likelihood), axis=0)
        self.RE = -np.log(self.BME) - self.NNCE

        expected_post_density = np.nanmean(np.log(self.post_density), axis=0)
        self.IE = -self.RE - expected_post_density

    @staticmethod
    def calculate_model_weight(data_array, order=0):
        """
        Calculates the model weights per column or per row.  Function sums all values along the given dimension (either
        row or column wise) and then divides by the number of models to get the score weight for each model.

        Args:
        ------------------
        :param data_array: Can be either a SxM (S=number of scores, M=number of models) to calculate for all scores at
         once or (1xM) to calculate for each score at a time
        :param order: determines whether the weighting is row-rise (order=1) or column-wise (order=0)
        -----------------
        :return: array with model weights, either SxM or 1xM depending on input. Each row sums to 1

        Note: If order=0, it means that each row corresponds to a different model, and each column is a different 'true
        measurement' data set.
        """
        if data_array.ndim == 1:
            score_array = data_array[np.newaxis, :]

        if order == 0:
            s_sum = np.nansum(data_array, axis=0)  # sum all values for each score (each column)
        else:
            s_sum = np.nansum(data_array, axis=1)  # sum all values for each score (row-wise)

        # calculate the weight for each model, for each score
        weights = np.divide(data_array, s_sum)

        return weights

    def run_bayes_inference(self):
        """
        Main code that runs the different BMS steps: calculates likelihood, rejection sampling and score calculations
        """
        logger.debug(f"Run Bayes Inference procedure for BMJ, with M_k = {self.model_name}")
        # 1. Calculate likelihood
        self.likelihood = self.calculate_likelihood_multiple_means()

        # 2. Rejection sampling
        self.rejection_sampling_vectorize()

        # 3. Calculate scores
        self.calculate_scores()


class BMJ:
    """
    Class runs Bayesian model similarity analysis for a set of competing models.

    Args:
    num_data_points: integer with number of data points being analyzed.
    mc_size_data_generating_model: int, with number of synthetic data sets for each generating model
    list_model_runs: list with BayesInference class instances. Each instance is a competing model, and contains
                    model name, prior, and model outputs.
    list_data_generating_models: list with MeasurementData instances. Each instance is a data generating model and
                                contains the synthetic measurement data and measurement error.


    Attributes:
    ------------------------------------
    self.num_data_points = int, with number of data points being analyzed
    self.num_nd = int, with number of synthetically generated measurement data sets for each generating model
    self.complete_model_runs = list_model_runs
    self.complete_generating_models = list_data_generating_models

    self.num_models = int, number of models being analyzed.

    self.BME_CM = np.array with confusion matrix for bme model weights
    self.NNCE_CM = np.array with confusion matrix for non-normalized cross entropy scores
    self.RE_CM = np.array with confusion matrix for relative entropy scores
    self.IE_CM = np.array with confusion matrix for information entropy scores
    self.logBME_CM = np.array with confusion matrix for -log(bme) scores

    * Normalized confusion matrices: all columns are normalized based on the diagonal value, to quantify the deviation
    from the self-identification (normalization is done for each M_j model run, and then the values are averaged)

    self.BME_norm = np.array with normalized confusion matrix for -log(BME) values
    self.NNCE_norm = np.array with normalized confusion matrix for NNCE values
    self.RE_norm = np.array with normalized confusion matrix for RE values
    self.IE_norm = np.array with normalized confusion matrix for IE values
    """

    def __init__(self, num_data_points, mc_size_data_generating_model, list_model_runs, list_data_generating_models):
        self.num_data_points = num_data_points
        self.num_nd = mc_size_data_generating_model
        self.complete_model_runs = list_model_runs
        self.complete_generating_models = list_data_generating_models

        self.num_models = len(self.complete_model_runs)

        self.BME_CM = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)
        self.NNCE_CM = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)
        self.RE_CM = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)
        self.IE_CM = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)

        self.logBME_CM = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)

        self.BME_norm = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)
        self.NNCE_norm = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)
        self.RE_norm = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)
        self.IE_norm = np.full((len(self.complete_generating_models), len(self.complete_model_runs)), 0.0)

        logger.debug("Generated BMJ instance")

    @staticmethod
    def synthetic_measurement_noise(measurement_values, measurement_error):
        """
        Function adds noise to synthetic measurement values, based on the measurement error. The noise is generated by
        a Gaussian distribution with mean of 0 and standard deviation equal to measurement error.

        Args:
        ------------------------------------
        :param measurement_values: np.array with synthetic measurement values. Size Nd x Np, where Nd is the number of
        synthetical sets generated and Np is the total number of measurement points for the given model.
        :param measurement_error: np.array with measurement error values for each data points. Size 1 x Np

        :return: np.array with modified synthetic measurement values, with noise.
        """
        # Check that error array is 1D
        if measurement_error.ndim == 2:
            if measurement_error.shape[0] == 1:
                measurement_error = measurement_error[0, :]
            else:
                measurement_error = measurement_error[:, 0]

        values_with_noise = np.full((measurement_values.shape[0], measurement_values.shape[1]), 0.0)
        for e, error in enumerate(measurement_error):
            noise = np.random.normal(loc=0, scale=error, size=measurement_values.shape[0])
            values_with_noise[:, e] = measurement_values[:, e] + noise

        return values_with_noise

    def run_bmj(self):
        """
        Function runs Bayesian model similarity analysis for the total number of available measurement data

        Notes:
        * Function has two main loops, one through each data generating model (M_j) and a nested loop for each
        competing model (M_k). The second nested loop includes all Nd realization of model M_j.
        * All calculations, weighting, normalizations, etc. are done for each run of model M_j individually and
        then, as a last step, the values are averaged for each M_j realization to get the averaged confusion matrices
        input for each column (M_j).
        """
        for j in range(len(self.complete_generating_models)):
            logger.debug(f'Running BMJ analysis for M_j = {self.complete_model_runs[j].model_name}.')
            OM_j = self.complete_generating_models[j]

            bme_mj = np.full((len(self.complete_model_runs), self.num_nd), 0.0)
            nnce_mj = np.full((len(self.complete_model_runs), self.num_nd), 0.0)
            re_mj = np.full((len(self.complete_model_runs), self.num_nd), 0.0)
            ie_mj = np.full((len(self.complete_model_runs), self.num_nd), 0.0)

            d = f'Running BMJ for M_j = {j}'
            for k in trange(len(self.complete_model_runs), desc=d, unit='model'):
                comparable_models = True
                model = self.complete_model_runs[k]
                # Determine if measurement values need to be filtered and generate BI_BMJ instance
                if OM_j.meas_type is not None:
                    mt_k = np.unique(model.measurement_data.meas_type)
                    mt_j = np.unique(OM_j.meas_type)

                    # (if models have same number of meas types but different types) or (mk has less types than mj)
                    if (len(mt_k) == len(mt_j) and (mt_k != mt_j).all()) or len(mt_k) < len(mt_j):
                        comparable_models = False
                        f_loc, f_vals, f_errs = OM_j.filter_by_type(model.measurement_data.meas_type)
                        f_OM = MeasurementData(f_loc, f_vals, f_errs)
                        f_OM.generate_cov_matrix()
                        f_OM.meas_type = model.measurement_data.meas_type
                        BI = BayesInferenceBMJ(f_OM, k + 1)
                    else:
                        BI = BayesInferenceBMJ(OM_j, k + 1)
                else:
                    BI = BayesInferenceBMJ(OM_j, k + 1)

                # Assign attributes
                BI.model_name = model.model_name
                BI.prior_density = model.prior_density

                # Assign output
                # Filter if M_j has more meas data than Mk
                if BI.measurement_data.meas_type is not None and len(mt_k) > len(mt_j):
                    BI.output = MeasurementData.filter_by_input(model.output, model.measurement_data.meas_type,
                                                                np.unique(mt_j))
                else:
                    BI.output = model.output

                # Run bayes inference
                BI.run_bayes_inference()
                if comparable_models:  # If models have the same number and type of measurement data points
                    bme_mj[k, :] = BI.BME
                    nnce_mj[k, :] = BI.NNCE
                else:
                    bme_mj[k, :] = np.full_like(BI.BME, np.nan)
                    nnce_mj[k, :] = np.full_like(BI.NNCE, np.nan)
                re_mj[k, :] = BI.RE
                ie_mj[k, :] = BI.IE

            # BME weights
            self.BME_CM[:, j] = np.mean(BI.calculate_model_weight(bme_mj), axis=1)

            # score values: average for all runs of model M_j
            log_bme_mj = -np.log(bme_mj)
            self.logBME_CM[:, j] = np.mean(log_bme_mj, axis=1)
            self.NNCE_CM[:, j] = np.mean(nnce_mj, axis=1)
            self.RE_CM[:, j] = np.mean(re_mj, axis=1)
            self.IE_CM[:, j] = np.mean(ie_mj, axis=1)

            # score normalization and subsequent averaging for all runs of model M_j
            self.BME_norm[:, j] = np.mean(np.divide(log_bme_mj, log_bme_mj[j, :]), axis=1)
            self.NNCE_norm[:, j] = np.mean(np.divide(nnce_mj, nnce_mj[j, :]), axis=1)
            self.RE_norm[:, j] = np.mean(np.divide(re_mj, re_mj[j, :]), axis=1)
            self.IE_norm[:, j] = np.mean(np.divide(ie_mj, ie_mj[j, :]), axis=1)

            # remove infinite values
            self.BME_norm[:, j] = np.where(self.BME_norm[:, j] == np.inf, np.nan, self.BME_norm[:, j])
            self.logBME_CM[:, j] = np.where(self.logBME_CM[:, j] == np.inf, np.nan, self.logBME_CM[:, j])
