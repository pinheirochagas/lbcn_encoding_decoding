# Some usefull tools by Pedro Pinheiro-Chagas
import pandas as pd
import numpy as np


def combine_variables_df(variables, labels):
    """
    Takes bar and does some things to it.
    """
    data_cat = []
    labels_cat = []
    for i in range(0, len(variables)):
        data_cat = np.append(data_cat, variables[0])
        labels_tmp = [labels[i]] * len(variables[0])
        labels_cat.extend(labels_tmp)
    data = pd.DataFrame(data_cat, columns={'data'})
    labels = pd.DataFrame(labels_cat, columns={'labels'})
    df = pd.concat([data, labels], axis=1)
    return df


def corr_vec(X, Y, ax=0):
    """
    Vectorized Pearson's correlation
    X and Y can be 2D
    Default axis is rows (0)

    Parameters
    ----------
    X : numpy array up to 2D
    Y : numpy array up to 2D
    ax : axis to perform the operation

    Returns
    -------
    corr_coef: Pearson's correlation coefficient r for each element in axis

    Notes
    -----
    Thanks to Michael Eickenberg
    """
    Xs = X - X.mean(axis=ax)
    Xs /= np.linalg.norm(Xs, axis=ax) + 1e-9
    Ys = Y - Y.mean(axis=ax)
    Ys /= np.linalg.norm(Ys, axis=ax) + 1e-9
    corr_coef = np.einsum("ij, ij -> j", Xs, Ys)
    return corr_coef
