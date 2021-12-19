"""
Module contains functions related to saving and exporting results to .txt files, and additional functions
"""

from config import *

score_list_uncertainty = ["-log(BME)", "NNCE", "RE", "IE"]


def save_results_txt(data_array, row_names, save_name, col_names=None, suffix=None):
    """
    Function saves input data array to .txt files

    Args:
    :param data_array: np.array (2D or 3D) with score results. Each row in 2D array is a different model, and each
    column is a score value, in the order log(BME), ELPD, RE, IE
    :param row_names: np.array or list with row titles
    :param save_name: str with name with which to save the resulting .txt file
    :param col_names: optional, np.array or list with column titles. If none is give, the function assumes it should
    use the score names
    :param suffix: optional, list or np.array to be used when input array is a 3D array, name with which to save each
    2D array will have the corresponding value added at the end of the name (to ID it)
    """
    if col_names is None:
        col_names = score_list_uncertainty

    df_c1 = pd.DataFrame(data=row_names, columns=['Model'])

    if data_array.ndim == 2:
        df = pd.DataFrame(data=data_array, columns=[col_names])
        df = pd.concat([df_c1, df], axis=1)

        df.to_csv(save_name, index=False, sep='\t')
    else:
        for k in range(data_array.shape[0]):
            da = data_array[k, :, :]

            df = pd.DataFrame(data=da, columns=[col_names])
            df = pd.concat([df_c1, df], axis=1)

            sn = os.path.join(save_name, f'BMS_multiple_points_n{str(suffix[k])}.txt')
            df.to_csv(sn, index=False, sep='\t')
