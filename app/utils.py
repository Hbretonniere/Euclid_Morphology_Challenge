import numpy as np
import streamlit as st
from astropy.io import ascii

def compute_snr_weights(mag_bins, order=3):
    TU_mags = np.linspace(16, 26, 11)
    TU_snrs = np.array(
        [7.50250212,
        6.79360416,
        6.09274036,
        5.39695112,
        4.70327681,
        4.00875781,
        3.31043451,
        2.60534727,
        1.89053648,
        1.16304252,
        0.41990577
        ]
    )

    coefs = np.polyfit(TU_mags, TU_snrs, order)[::-1]
    weights = np.zeros_like(mag_bins)
    for i, coef in enumerate(coefs):
        weights += coef * mag_bins ** i
    return weights


def compute_score(
    means,
    sigmas,
    outliers,
    completeness,
    weights,
    factors=[2.5, 2.5, 2.5, 1],
    linear_SNR_weight=False,
):
    if linear_SNR_weight:
        weights = np.ones(len(weights))
    k_m, k_s, k_out, k_c = factors
    means = np.abs(means)
    score = k_c * (1 - completeness) + np.sum(
        weights * (
            k_m * np.array(means) + k_s * (np.array(sigmas)) + k_out * np.array(outliers)
        )
    )
    return score


def compute_disp(bias, quantile=0.68):
    return np.nanquantile(np.abs(bias) - np.nanmedian(bias), quantile)


def chose_mode(param):
    if param in ["re", "mag", "reb", "red"]:
        mode = "relative"
    elif param in ['n', 'BulgeSersic']:
        mode = 'log'
    else:
        mode = "absolute"

    return mode


def sub_cat(cat, x_axis, x_bins, bin):

    indices = np.where((cat[x_axis] > x_bins[bin]) & (cat[x_axis] < x_bins[bin + 1]))[0]
    nb_gal = len(cat)
    fraction = len(indices) / nb_gal
    return cat.iloc[indices], fraction


# @st.cache
def sub_cat2d(cat, x_axis, x_bins, y_axis, y_bins, xbin, ybin):
    indices = np.where(
        (cat[x_axis] > x_bins[xbin])
        & (cat[x_axis] < x_bins[xbin + 1])
        & (cat[y_axis] > y_bins[ybin])
        & (cat[y_axis] < y_bins[ybin + 1])
    )[0]

    nb_gal = 275664
    fraction = len(indices) / nb_gal

    return cat.iloc[indices], fraction


def compute_error_bin(cat, code, param, mode="absolute", abs=True):
    """
    Compute the individual biases (small b) for a certain parameter, code and catalogue (or sub-catalogue)

    Parameters
    ----------
    cat : pandas dataframe
        The input catalogue for computing the biases.
    
    code: string
        The name of the software package you want to study.

    param: string
        The name of the parameter you want to study.
    
    mode: string
        Define the type of bias, between normal and relative, If "relative",
        compute the relative bias, i.e divide by the true value of the parameter.
        TODO: change this to boolean

    abs: boolean
        If True, compute the absolute value bias instead of the normal one.


    Return
    ----------
    error : dataframe
        The dataframe containing all the individual biases of the input catalogue.

    """

    true = f"True{param}"
    pred = f"Pred{param}_{code}"
    param_pred = cat[pred]
    param_true = cat[true]

    error = param_pred - param_true

    if abs:
        error = np.abs(error)
    
    if mode == 'log':
        error = np.log10(param_pred) - np.log10(param_true)

    if mode == "relative":
        error /= param_true

    return error


def find_outliers(cat, code, param, mode, outlier_limit, abs=True):
    """
    Compute the indices of the bad fit or outliers.

    Parameters
    ----------
    cat : pandas dataframe
        The input catalogue for computing the biases.
    
    code: string
        The name of the software package you want to study.

    param: string
        The name of the parameter you want to study.
    
    mode: string
        Define the type of bias, between normal and relative, If "relative",
        compute the relative bias, i.e divide by the true value of the parameter.
        TODO: change this to boolean

    outlier_limit:
        The threshold (b_p in the paper) to define what is a bad fit.
     
    abs: boolean
        If True, compute the absolute value bias instead of the normal one.


    Return
    ----------
    error : dataframe
        The dataframe containing all indices considered as bad fits.

    """

    error = cat[f"Pred{param}_{code}"] - cat[f"True{param}"]

    if abs:
        error = np.abs(error)

    if mode == 'log':
        error = np.log10(cat[f"Pred{param}_{code}"]) - np.log10(cat[f"True{param}"])

    if mode == "relative":
        relative_error = error / cat[f"True{param}"]
        indices_out = np.where(np.abs(relative_error) > outlier_limit)[0]

    if mode in ["absolute", "log"]:
        indices_out = np.where(np.abs(error) > outlier_limit)[0]

    return indices_out


@st.cache
def compute_summary(
    cat,
    params,
    codes,
    x_bins,
    outlier_limit=None,
    dataset="single_sersic",
    factors=[1, 2.1, 2.1, 2.1],
    x_axis="Truemag",
    abs=False,
    linear_SNR_weight=False,
):
    """
    Compute summary statistics of different parameters, for given software.

    Parameters
    ----------
    cat : pandas dataframe
        The input catalogue to compute the biases.

    params : list of string
        The names of the parameters you want to study.
        Should match the names of the catalogues' columns.

    codes : list of strings
        The different codes you want to study.
        Should be one of the following : SE++, profit, gala, deepleg, metryka.

    x_bins : list of float
        The list of bins of the x-axis (magnitude or b/t) for the study.
        Stop at x_bins[-1].

    outlier_limit : float
        The value of the outlier threshold. Objects which have an error bigger
        than this number will be rejected during the study.
        If None, no outliers are removed.
                        
    dataset : string.
        The type of simulation you want to study. Must be "single_sersic",
         "double_sersic", or "realistic".

    Returns
    -------
    code_means : dict
        The different means errors. There is a key by parameter name,
        and a key by code. in each, the list of mean error by bins of magnitude.
        For example, code_means['gala']['re'] will give you
        the list of mean errors by bin of magnitude made by Galapagos-2
        on the fitting of the radius.

    code_stds : dict
        Same but for stds.

    code_outs : dict
        Same but for the fraction of outliers.

    Example
    -------
    >>>  compute_summary(cat, ['mag', 're', 'q'], ['gala', 'profit'], np.linspace(18, 25, 2), outlier_limit=0.5, dataset='single_sersic')
    """

    # Initialize the outputs

    params_means, params_stds, params_outs, params_scores, params_scores_global = (
        {},
        {},
        {},
        {},
        {},
    )
    completenesses = {
        "single_sersic": {
            "SE++": 0.95,
            "gala": 0.88,
            "profit": 0.99,
            "deepleg": 0.89,
            "metryka": 0.82,
        },
        "double_sersic": {"SE++": 0.97, "gala": 0.94, "profit": 1.00, "deepleg": 0.98},
        "multiband": {"SE++": 0.95, "gala": 0.93, "profit": 0.98, "deepleg": 0.98},
        "realistic": {"SE++": 0.85, "gala": 0.71, "profit": 0.92},
    }

    #  Loop through the parameters you want to study
    for param in params:

        # Create the dictionnary corresponding to the parameter
        (
            params_means[param],
            params_stds[param],
            params_outs[param],
            params_scores[param],
            params_scores_global[param],
        ) = ({}, {}, {}, {}, {})

        mode = chose_mode(param)
        # Loop through the codes you want to study

        for code in codes:
            # Create the list for the mean and stds of the particular code and param (always double component and then remove the disks if its ss)
            means = np.zeros((2, len(x_bins) - 1))
            stds = np.zeros_like(means)
            outs = np.zeros_like(means)
            scores = np.zeros_like(means)
            fracgals = []
            gal_bin = []
            # Loop through the magnitude bins'''
            for i in range(len(x_bins) - 1):
                cat_bin, fraction = sub_cat(cat, x_axis, x_bins, i)
                fracgals.append(fraction)
                gal_bin.append(len(cat_bin))
                error = compute_error_bin(cat_bin, code, param, mode=mode, abs=abs)
                

                if len(error) == 0:
                    raise RuntimeError(
                        f"ah ! No galaxy in the bin of magnitude {x_bins[i]-x_bins[i+1]:.2f}, try reducing the magnitude range."
                    )

                outliers = find_outliers(
                    cat_bin, code, param, mode, outlier_limit, abs=abs
                )

                means[0, i] = np.nanmedian(error)
                stds[0, i] = compute_disp(error)
                outs[0, i] = (
                    len(cat_bin) - len(error.drop(error.index[outliers]))
                ) / len(cat_bin)
                scores[0, i] = (
                    factors[0] * means[0, i] + factors[1] * stds[0, i] + factors[2] * outs[0, i]
                )

            center_bins = (x_bins[1:] + x_bins[:-1]) * 0.5
            snr_weights = compute_snr_weights(center_bins)
            weights = gal_bin[: len(x_bins) - 1] * snr_weights
            weights /= np.sum(weights)
            weights = weights[: len(x_bins)]

            means = means[0]
            stds = stds[0]
            outs = outs[0]
            scores = scores[0]
            completeness = completenesses[dataset][code]
            params_scores_global[param][code] = compute_score(
                means,
                stds,
                outs,
                completeness,
                weights,
                factors=factors,
                linear_SNR_weight=linear_SNR_weight,
            )

            # At the end of the loop in the mag bins, add the list to the good code and param key of the dictionnary
            params_means[param][code] = means
            params_stds[param][code] = stds
            params_outs[param][code] = outs
            params_scores[param][code] = scores
            

    return [params_means, params_stds, params_outs, params_scores, params_scores_global]


def compute_summary2D(
    cat,
    params,
    codes,
    x_bins,
    y_bins,
    x_axis="Truemag",
    y_axis="Truebt",
    abs=True,
):
    """
    Compute the Means, the Standard Deviation and the fraction of outliers of errors in the fitting of different parameters, for different codes.

    Inputs:
            - cat : pandas data-frame
                The catalogue containing all the galaxies of all the codes

            - params : list of strings. 
                The names of the parameters you want to study.
                Should match the names of the homogenized catalogues.

            - codes : list of strings. 
                The different codes you want to study.
                should be one of the following : SE++, profit, gala, deepleg, metryka.

            - x_bins : list of float. 
                The list of bins of the x-axis (magnitude or b/t) for the study. Stop at x_bins[-1].

            - dataset : string. 
                The type of fit you want to study. Must be "single_sersic", "double_sersic", or "realistic".

    Outputs :
            - code_means : dictionnary. 
                The different means errors. There is a key by parameter name,
                and a key by code. in each, the list of mean error by bins of magnitude.
                For example, code_means['gala']['re'] will give you the list of mean errors
                by bin of magnitude
                made by galapagos on the fitting of the radius.

            - code_stds : dictionnary. 
                Same but for stds.

            - code_outs : dictionnary.
                Same but for the fraction of outliers.

    Example of use :

          compute_summary(cat, ['mag', 're', 'q'], ['gala', 'profit'], np.linspace(18, 25, 2), outlier_limit=0.5, dataset='single_sersic')
    """

    # Initialize the outputs
    params_means, params_stds, params_outs, params_scores = {}, {}, {}, {}
    write = 0
    for param in params:
        (
            params_means[param],
            params_stds[param],
            params_outs[param],
            params_scores[param],
        ) = ({}, {}, {}, {})
        mode = chose_mode(param)

        # Loop through the codes you want to study
        for c, code in enumerate(codes):

            # Create the list for the mean and stds of the particular code and param
            means = np.zeros((2, len(x_bins) - 1, len(y_bins) - 1))
            stds = np.zeros_like(means)
            outs = np.zeros_like(means)
            fracgals = []

            # Loop through the magnitude bins'''
            for i in range(len(x_bins) - 1):
                for j in range(len(y_bins) - 1):
                    # try:
                        cat_bin, fraction = sub_cat2d(
                            cat, x_axis, x_bins, y_axis, y_bins, i, j
                        )
                        fracgals.append(fraction)

                        abs_error_b = compute_error_bin(
                            cat_bin, code, param + "b", mode=mode, abs=True
                        )

                        abs_error_d = compute_error_bin(
                            cat_bin, code, param + "d", mode=mode, abs=True
                        )

                        classic_error_b = compute_error_bin(
                            cat_bin, code, param + "b", mode=mode, abs=False
                        )

                        classic_error_d = compute_error_bin(
                            cat_bin, code, param + "d", mode=mode, abs=False
                        )

                        means[0, i, j] = np.nanmedian(abs_error_b)
                        means[1, i, j] = np.nanmedian(abs_error_d)

                        stds[0, i, j] = compute_disp(classic_error_b)
                        stds[1, i, j] = compute_disp(classic_error_d)

                        outb = (len(cat_bin) - len(classic_error_d)) / len(cat_bin)
                        outd = (len(cat_bin) - len(classic_error_d)) / len(cat_bin)
                        outs[0, i, j] = outb
                        outs[1, i, j] = outd

                params_means[param][code] = means
                params_stds[param][code] = stds
                params_outs[param][code] = outs

    return [params_means, params_stds, params_outs]


def format_func(value):
    return np.round(value, 2)


def compute_error_prediction(cat, params, codes, x_bins, nb_bins):
    """
    Compute the fraction of objects with a well calibrated uncertainty. 

    Inputs:
            - cat : pandas data-frame
                The catalogue containing all the galaxies of all the codes

            - params : list of strings
                The names of the parameters you want to study.
                Should match the names of the homogenized catalogues.

            - codes : list of strings
                The different codes you want to study.
                should be one of the following : SE++, profit, gala, deepleg, metryka.

            - x_bins : list of float
                The list of bins of the x-axis (magnitude or b/t) for the study. Stop at x_bins[-1].

    Outputs :
            - calib_mag : dictionnary
                The fraction of objects with a well calibrated uncertainty, for each code, parameter and
                bin of magnitude.

    Example of use :

          compute_summary(cat, ['mag', 're', 'q'], ['gala', 'profit'], np.linspace(18, 25, 2), outlier_limit=0.5, dataset='single_sersic')
    """

    calib_mag = {}
    for code in codes:
        calib_mag[code] = {}
        for param in params:
            ins = []
            calibs = []
            calib_mag[param] = np.zeros(nb_bins)
            for i in range(len(x_bins) - 1):
                bin_cat, _ = sub_cat(cat, "Truemag", x_bins, i)
                in_ = []
                for i, err in enumerate(bin_cat[f"Pred{param}err_{code}"]):
                    predm = bin_cat[f"Pred{param}_{code}"].iloc[i]
                    truem = bin_cat[f"True{param}"].iloc[i]
                    if (truem <= predm + err) & (truem > predm - err):
                        in_.append(i)

                   
                ins.append(len(in_))
                all = len(np.where((cat['Truemag'] > x_bins[0]) & (cat['Truemag'] < x_bins[-1]))[0])
                fraction = len(in_) / len(bin_cat[f"Pred{param}err_{code}"])
                calibs.append(fraction)
            calibs.append(sum(ins) / all)
            calib_mag[code][param] = calibs

    return calib_mag


@st.cache
def compute_bt_multiband(
    cat,
    codes,
    bands,
    outlier_limit=None,
    factors=[1, 1, 1],
    x_axis="Truemag",
    abs=True
):
    """
    Compute summary statistics for the bulge-to-total ratio, 
    in the multi-band case.

    Parameters
    ----------
    cat : pandas data-frame
        The catalogue containing all the galaxies of all the codes

    codes : list of strings
        The different codes you want to study.
        Should be one of the following : SE++, profit, gala, deepleg, metryka.

    bands : list of string.
        The names of the bands you want to study.

    x_bins : list of float
        The list of bins of the x-axis (magnitude or b/t) for the study. Stop at x_bins[-1].

    outlier_limit : float. 
        The value of the outlier threshold. Objects which have an error bigger than this number
        will be rejected during the study. If None, no outliers are removed.

    dataset : string
        The type of fit you want to study. Must be "single_sersic", "double_sersic", or "realistic".

    Returns
    -------
    List of dictionaries:
    code_means : dict
        The different means errors. There is a key by band name,
        and a key by code. in each, the list of mean error by bins of magnitude.
        For example, code_means['gala']['re'] will give you the list of mean errors by bin of magnitude
        made by galapagos on the fitting of the radius.
    code_stds : dict
        Same but for stds.
    code_outs : dict
        Same but for the fraction of outliers.

    """

    # Initialize the outputs
    params_means, params_stds, params_outs, params_scores, params_scores_global = (
        {},
        {},
        {},
        {},
        {},
    )

    #  Loop through the parameters you want to study
    for code in codes:

        # Create the dictionnary corresponding to the parameter
        (
            params_means[code],
            params_stds[code],
            params_outs[code],
            params_scores[code],
            params_scores_global[code],
        ) = ({}, {}, {}, {}, {})

        mode = chose_mode('bt')
        # Loop through the codes you want to study
        x_bins = np.linspace(16, 26, 10)
        for band in bands:
            # Create the list for the mean and stds of the particular code and param (always double component and then remove the disks if its ss)
            means = np.zeros(len(x_bins-1))
            stds = np.zeros_like(means)
            outs = np.zeros_like(means)
            scores = np.zeros_like(means)
            fracgals = []
            gal_bin = []
            # Loop through the magnitude bins'''
            for i in range(len(x_bins) - 1):
                cat_bin, fraction = sub_cat(cat, x_axis, x_bins, i)
                fracgals.append(fraction)
                gal_bin.append(len(cat_bin))
                error = compute_error_bin(cat_bin, code, f'bt_{band}', mode=mode, abs=abs)

                outliers = find_outliers(
                    cat_bin, code, f'bt_{band}', mode, outlier_limit, abs=abs
                )
                means[i] = np.nanmedian(error)
                stds[i] = compute_disp(error)
                outs[i] = (
                    len(cat_bin) - len(error.drop(error.index[outliers]))
                ) / len(cat_bin)
                scores[i] = (
                    factors[0] * means[i] + factors[1] * stds[i] + factors[2] * outs[i]
                )

            snr_weights = compute_snr_weights(x_bins[:-1])
            weights = gal_bin[: len(x_bins) - 1] * snr_weights
            weights /= np.sum(weights)
            weights = weights[: len(x_bins)]

            # At the end of the loop in the mag bins, add the list to the good code and param key of the dictionnary
            params_means[code][band] = means
            params_stds[code][band] = stds
            params_outs[code][band] = outs
            params_scores[code][band] = means

    return [params_means, params_stds, params_outs, params_scores, params_scores_global]

def define_y_axis_slider(metric, param):
    mins = []
    maxs = []
    codes = metric[param].keys()
    for code in codes:
        mins.append(np.min(metric[param][code]))
        maxs.append(np.max(metric[param][code]))
        min = np.min(mins)
        max = np.max(maxs)
    return min, max

def get_x_axis_range(dataset, x_axis, demo):
    if x_axis == "Truemag":
        if dataset == "realistic":
            min_x = 21.0 if demo else 20.0
        else:
            min_x = 19.0 if demo else 14.0
        x_button_name = "VIS True magnitude range"
        min_val, max_val, step = 18.0, 26.0, 0.5
        values = (min_x, 25.3)
        x_min, x_max = st.slider(
            x_button_name,
            min_value=min_val,
            max_value=max_val,
            value=values,
            step=step,
        )

    elif x_axis in ["Truere", "Truered"]:
        x_button_name = "True Radius range"
        options = np.around(np.logspace(-0.8, 0.1, 10), 2)
        x_min, x_max = st.select_slider(
            x_button_name,
            value=(options[0], options[-1]),
            options=options
        )

    elif x_axis == "Truez":
        min_x = 0.3 if demo else 0.
        x_button_name = "VIS True redshift range"
        min_val, max_val, step = 0., 6., 0.1
        values = (min_x, 3.5)
        x_min, x_max = st.slider(
            x_button_name,
            min_value=min_val,
            max_value=max_val,
            value=values,
            step=step,
        )

    elif x_axis == "Truebt":
        x_button_name = "True b/t range"
        options = np.around(np.logspace(-2, 0, 20), 2)
        values = (0., 1.)
        x_min, x_max = st.select_slider(
            x_button_name,
            value=(options[0], options[-1]),
            options=options
        )
    
    elif x_axis == "Truen":
        x_button_name = "True Sérsic Index"
        min_val, max_val, step = 0.3, 6., 0.1
        values = (0.3, 5.7)
        x_min, x_max = st.slider(
            x_button_name,
            min_value=min_val,
            max_value=max_val,
            value=values,
            step=step,
        )
    xs = [x_min, x_max]
    return xs
