from config import *
from log import *


class MeasurementData:
    """
    Class saves information regarding to measurement data, be it real measurement data or synthetically generated
    measurement data

    Input parameters:
    ------------------------
    point_loc: np.array with location (x, y, z, t) of each measurement data point
    values: np.array with measurement values
    error: np.array with error value for each measurement

    Class attributes:
    -----------------------
    self.loc: np.array with point_loc information
    self.meas_values: np.array with values data
    self.error: np.array with error data for each measurement point

    self.num_meas = int, number of available measurement points
    self.R = np.array, covariance matrix of measurement data
    self.meas_type: np.array of strings, with type of measurement data for each point (optional)
    """
    def __init__(self, point_loc, values, error):
        self.loc = point_loc
        self.meas_values = values
        self.error = error

        self.num_meas = self.get_num_meas_points()
        self.cov_matrix = np.array([])
        self.meas_type = None

        # Run check functions
        self.check_matrix_size()

    def get_num_meas_points(self):
        """
        Function calculates the number of measurement data points available

        :return: int, number of data points
        """
        num_mp = np.max(self.loc.shape)
        return num_mp

    def check_matrix_size(self):
        """
        Function checks if each measurement value and error correspond to a different column in the array. Each row
        should correspond to a set of measurement data.
        """
        # Measurement values:
        if self.num_meas.ndim == 1:
            self.num_meas = self.num_meas.reshape(1, len(self.num_meas))
        else:
            if self.meas_values.shape[1] != self.num_meas:
                self.meas_values = self.meas_values.T
        # Measurement error:
        if self.error.ndim > 1 and self.error.shape[1] != self.num_meas:
            self.error = self.error.T
        # Measurement loc
        if self.loc.ndim > 1 and self.loc.shape[1] != self.num_meas:
            self.loc = self.loc.T

    def generate_cov_matrix(self):
        """
        Function generates or fills the diagonal covariance matrix of independent measurement data, where the diagonal
        is the variance of the data (error^2) and all other entries are 0.
        """
        cov_matrix = np.full((self.num_meas, self.num_meas), 0.0)

        try:
            row, col = np.diag_indices(cov_matrix.shape[0])
            cov_matrix[row, col] = np.power(self.error, 2)
        except ValueError as e:
            logger.exception("There should be one measurement error per measurement value ")
            sys.exit()
        else:
            self.cov_matrix = cov_matrix

    def generate_meas_type(self, type_list, num_dp):
        """
        Function generates the meas_type array from a list that contains the type of measurement points to consider.

        Args
        -------------------------
        :param type_list: list with strings, which contain the type of measurement data
        :param num_dp: int, with number of data points (or how many times to repeat the measurement type value)
        """

        self.meas_type = np.repeat(np.array(type_list), num_dp)

    def filter_by_type(self, type_list):
        """
        Function filters the measurement data according to measurement type.

        Args:
        --------------------------
        :param type_list: list of strings, with the type of measurement data type to include in the filtered values.

        :return: np.arrays with filtered measurement values and filtered error values
        """
        mask = np.isin(self.meas_type, type_list)
        index = np.where(mask)[0]

        if self.loc.ndim == 1:
            filtered_locs = self.loc[index]
        else:
            filtered_locs = np.take(self.loc, index, axis=1)
        filtered_vals = self.meas_values[:, index]
        filtered_errs = self.error[:, index]

        if filtered_vals.size == 0:
            m = "There are no coinciding measurement data types between the measurement data instance and the input " \
                "type list. No filtered values were returned."
            logger.warning(m)
            sys.exit()
        else:
            return filtered_locs, filtered_vals, filtered_errs

    def filter_by_index(self, index_list):
        """
        Function filters the measurement data according to measurement type.

        Args:
        --------------------------
        :param index_list: list of int, with the index of the columns to be filtered (remove all others).

        :return: np.arrays with filtered measurement values and filtered error values
        """

        if self.loc.ndim == 1:
            filtered_locs = np.take(self.loc, index_list)
        else:
            filtered_locs = np.take(self.loc, index_list, axis=1)

        filtered_vals = self.meas_values[:, index_list]

        if self.error.ndim == 1:
            filtered_errs = np.take(self.error, index_list)
        else:
            filtered_errs = self.error[:, index_list]

        if filtered_vals.size == 0:
            m = "There are no coinciding measurement data types between the measurement data instance and the input " \
                "type list. No filtered values were returned."
            logger.warning(m)
            sys.exit()
        else:
            return filtered_locs, filtered_vals, filtered_errs

    @staticmethod
    def filter_by_input(a, filter_criteria, filter_list):
        """
        Function filters the input array 'a' according to the indexes where a filter criteria array has values contained
        in the filter list

        Args:
        -----------------------------
        :param a: np.array, to filter (must have the same number of columns as the filter_criteria array)
        :param filter_criteria: np.array to compare with the filter_list values
        :param filter_list: np.array or list with filtering criteria

        :return: np.array, with filtered a array

        Note: The function gets the indexes where the filter_criteria array contains any of the values in the
        filter_list. These indexes are 'column' indexes and so all rows with said column indexes are filtered and saved
        to a new array.
        """
        mask = np.isin(filter_criteria, filter_list)
        index = np.where(mask)[0]

        if a.ndim == 1:
            filtered_a = a[index]
        else:
            filtered_a = a[:, index]

        if filtered_a.size == 0:
            m = "There are no coinciding measurement data types between the measurement data instance and the input " \
                "type list. No filtered values were returned."
            logger.warning(m)
            sys.exit()
        else:
            return filtered_a





