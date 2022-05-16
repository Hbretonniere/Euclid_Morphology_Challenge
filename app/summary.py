import numpy as np
import pandas as pd
import streamlit as st

from app.utils import (
    compute_summary,
    compute_summary2D,
    compute_error_prediction,
    compute_bt_multiband)
from app.params import LABELS
from app.plots import (
    summary_plot,
    summary_plot2D,
    trumpet_plot,
    plot_score,
    plot_error_prediction,
    bt_multiband_plot
)


def summary(
    df,
    dataset,
    params,
    codes,
    x_range,
    nb_bins,
    outlier_threshold,
    x_axis,
    score_factors,
    limit_y_axis=True,
    abs=False,
    show_scores=False,
    bd_together=False,
):

    """
    Compute and plot the Summary Figure representing the different metrics (Bias, Dispersion and Outlier fraction)
    for different codes and parameters.

    Parameters
    ----------
    df : panda dataframe
        The catalogue of the dataset, with all codes and parameters

    dataset : string
        The name of the dataset, e.g. 'single_sersic'.

    params : list of string
        The parameters names you want to study.

    codes : list of string
        The codes names you want to study.

    x_range : list of float, length 2
        The min and max magnitudes you want to do your study on.

    nb_bins : int
        The number of bins to study.

    outlier_threshold : float
        The threshold to define what is considered as an outlier

    x_axis : string
        Define the parameter to plot the metrics against. e.g. "True magnitude"

    score_factors : list of floats, length of 4
        The weights applied to the different metrics.
        score_factors = [w_bias, w_dispersion, w_outliers, w_completeness]

    abs : boolean, optional, default False
        If True, the absolute value of the bias is taken

    show_scores : boolean, optional, default False
        If True, print the global score S for each parameter and code on the right of the plot, ordered per
        ranking.

    bd_together ; boolean, optional, default False
        If True, plot the disk and bulges component together in the same subplot

    Raise
    ----------
    If not enough galaxies, raise a corresponding error messsage

    Return
    ----------
    Nothing

    """

    description = st.expander("More information about this figure.")
    description.markdown(
            """
            Summary plot for the selected dataset. The different rows show the results for each
            selected parameter. The first column represents the mean bias per bin
            of the x-axis parameter (default to VIS magnitude, but can be changed in the left panel)
            $\mathcal{B}$ (see Eqs. 5 and 4 of the paper). The second
            column, shows the dispersion also as a function of magnitude, $\mathcal{D}$
            (see Eq. 6). The third column represents the fraction of outliers,
            $\mathcal{O}$. Each selected code is plotted with a different
            color as labelled
            """
        )

    x_min, x_max = x_range
    if x_axis in ['Truere', 'Truered']:
        x_bins = np.logspace(np.log10(x_min), np.log10(x_max), nb_bins + 1)
    elif x_axis == 'Truez' and dataset == 'realistic':
        st.markdown('# Not implemented yet')
    else:
        x_bins = np.linspace(x_min, x_max, nb_bins + 1)
    try:
        results = compute_summary(
            df,
            params,
            codes,
            x_bins,
            outlier_limit=outlier_threshold,
            dataset=dataset,
            factors=score_factors,
            x_axis=x_axis,
            abs=abs,
        )

        figure = summary_plot(
            results,
            x_bins,
            dataset,
            x_axis,
            limit_y_axis=limit_y_axis,
            show_scores=show_scores,
            labels=LABELS,
            bd_together=bd_together,
        )
        st.pyplot(figure)
    except RuntimeError as e:
        st.markdown(f"## {e}")
    return results


def summary2D(
    df, params, codes, x_range, n_bins_mag, bts, n_bins_bt, outlier_threshold
):
    """
    Compute and plot the 2D Summary Figure representing the Bias and Dispersion for different
    codes and parameters.

    Parameters
    ----------
    df : panda dataframe
        The catalogue of the dataset, with all codes and parameters

    params : list of string
        The parameters names you want to study.

    codes : list of string
        The codes names you want to study.

    x_range : list of float, length 2
        The min and max magnitudes you want to do your study on.

    nb_bins_mag : int
        The number of magnitude bins to study.

    nb_bins_bt : int
        The number of bulge over total flux ratio bins to study.

    outlier_threshold : float
        The threshold to define what is considered as an outlier

    Return
    ----------
    results: dictionary
        The dictionary needed to do the plot

    """

    description = st.expander("More information about this figure.")

    description.markdown(
        """
        Bias $mathcal{B}$ and dispersion $\matchcal{D}$ for the
        selected parameters as a function of apparent VIS
         magnitude and bulge-to-total light fraction. Each row
        shows a different code. For ProFit and Galapagos-2, the color
        scale is logarithmic. The left column shows the results for the
        bulge component, and the right one for the disk component. In
        each panel, the color of the squares is proportional to the mean
        bias of the objects in the magnitude and b/t bin, and the color
        of the dot inside each square indicates the dispersion (redder
        means lower dispersion). For most of the codes, we appreciate
        the expected behavior: both the bias and the dispersion in-
        crease at faint magnitudes and at small b/t for bulges, and big
        b/t for disks
        """
    )

    x_min, x_max = x_range
    mag_bins = np.linspace(x_min, x_max, n_bins_mag + 1)

    y_min, y_max = bts
    bt_bins = np.linspace(y_min, y_max, n_bins_bt + 1)
    for param in params:
        # try:
            results = compute_summary2D(
                df, params, codes, mag_bins, bt_bins
            )
            figure = summary_plot2D(codes, param, results, mag_bins, bt_bins, LABELS)
            st.pyplot(figure)
        # except ValueError:
        #     st.markdown(
        #         "## Not enough galaxies per bin, try to reduce the ranges or quit the demo mode."
        #     )
            # return 0
    return results


def trumpet(df, params, codes, x_axis, x_range, nb_bins, outlier_threshold, y_range):
    """
    Compute and plot the Summary Figure representing the different metrics (Bias, Dispersion and Outlier fraction)
    for different codes and parameters.

    Parameters
    ----------
    df : panda dataframe
        The catalogue of the dataset, with all codes and parameters

    params : list of string
        The parameters names you want to study.

    codes : list of string
        The codes names you want to study.

    x_range : list of float, length 2
        The min and max x-axis you want to do your study on.

    nb_bins : int
        The number of bins to study.

    outlier_threshold : float
        The threshold to define what is considered as an outlier.

    y_max : float
        The maximum (and minimum, -y_max) range for the y-axis.

    Return
    ----------
    Nothing

    """

    description = st.expander("More information about this figure.")
    description.markdown(
        """
        Scatter plots showing the recovery of the selected
        parameters measured from the selected dataset. Each panel shows a different
        code between the selected ones. The main plot of each panel shows the relative
        bias per galaxy as a function of apparent VIS magnitude, while we
        summarize the bias distribution as a histogram on the right.
        The opacity is proportional to the density; darker means more points.
        The blue solid line highlights a zero bias for reference,
        and the gray dash line represents the mean value of the bias
        for all magnitude bins together. The orange points indicate the running mean
        bias in bins of magnitude, with error bars representing the standard deviation after
        exclusion of outliers. Therefore this shows the ideal
        case and differs from the numbers in the summary plots which include
        all objects
        """
    )
    x_min, x_max = x_range
    for param in params:
        figure = trumpet_plot(
            df,
            codes,
            param,
            outlier_threshold,
            x_axis,
            [x_min, x_max],
            y_range,
            nb_bins,
            labels=LABELS,
            freq_scat=1,
        )

        st.pyplot(figure)


def score(
    df,
    dataset,
    params,
    codes,
    score_factors,
    x_range,
    nb_bins,
    outlier_threshold,
    x_axis,
    abs=False
):
    """
    Compute and plot the Summary score figure representing the global Score S
    for different codes and parameters.

    Parameters
    ----------
    df : panda dataframe
        The catalogue of the dataset, with all codes and parameters

    dataset : string
        The name of the dataset, e.g. 'single_sersic'.

    params : list of string
        The parameters names you want to study.

    codes : list of string
        The codes names you want to study.

    score_factors : list of floats, length of 4
        The weights applied to the different metrics.
        score_factors = [w_bias, w_dispersion, w_outliers, w_completeness]

    x_range : list of float, length 2
        The min and max x-axis you want to do your study on.

    nb_bins : int
        The number of bins to study.

    outlier_threshold : float
        The threshold to define what is considered as an outlier.

    x_axis : string
        Define the parameter to plot the metrics against. e.g. "True magnitude"

    abs : boolean, optional, default False
        If True, the absolute value of the bias is taken

    Return
    ----------
    Nothing

    """
    description = st.expander("More information about this figure.")

    description.markdown(
        "Summary of the global scores $mathcal{S} for the selected dataset, \
        The x-axis shows the different parameters. $\mu$ corresponds to the mean of the parameters. \
        The y-axis represents the corresponding global score S, for each parameter and code"
    )
    x_min, x_max = x_range
    x_bins = np.linspace(x_min, x_max, nb_bins + 1)
    column_left, column_right = st.columns(2)
    with column_left:

        k_m = st.slider(
            "Bias factor w_B", min_value=0.0, max_value=15.0, value=2.1, step=0.5
        )
        k_s = st.slider(
            "Dispersion factor w_D", min_value=0.0, max_value=15.0, value=2.1, step=0.5
        )
    with column_right:
        k_o = st.slider(
            "Outliers factor w_O", min_value=0.0, max_value=15.0, value=2.1, step=0.5
        )
        k_c = st.slider(
            "Completeness factor w_C",
            min_value=0.0,
            max_value=15.0,
            value=1.0,
            step=0.5,
        )
    linear_SNR_weight = st.checkbox("Remove the SNR weight on the global score")
    score_factors = [k_m, k_s, k_o, k_c]
    score = compute_summary(
        df,
        params,
        codes,
        x_bins,
        outlier_limit=outlier_threshold,
        dataset=dataset,
        factors=score_factors,
        x_axis=x_axis,
        abs=abs,
        linear_SNR_weight=linear_SNR_weight,
    )[-1]
    score = pd.DataFrame.from_dict(score)

    if dataset in ["single_sersic", "realistic"]:
        score.loc[:, "mu"] = score.mean(axis=1)
    else:
        bulge_params = np.zeros(len(codes))
        disk_params = np.zeros(len(codes))
        for param in score.columns:
            if param in ["q", "re"]:
                score.loc[:, f"Bulge {param}"] = score[param].explode()[::2].values
                score.loc[:, f"Disk {param}"] = score[param].explode()[1::2].values
                bulge_params = bulge_params + score[param].explode()[::2].values
                disk_params = disk_params + score[param].explode()[1::2].values
                score.drop(columns=[param], inplace=True)
            elif param == "bt":
                bulge_params += score[param]
                disk_params += score[param]
        score.loc[:, "Bulge mu"] = bulge_params / len(params)
        score.loc[:, "Disk mu"] = disk_params / len(params)
    score = score.round(2)

    figure = plot_score(score.transpose(), LABELS)
    st.pyplot(figure)
    st.dataframe(score)


def error_calibration(df, dataset, params, codes, x_range, nb_bins):
    """
    Compute and plot the error calibration Figure

    Parameters
    ----------
    df : panda dataframe
        The catalogue of the dataset, with all codes and parameters

    dataset : string
        The name of the dataset, e.g. 'single_sersic'.

    params : list of string
        The parameters names you want to study.

    codes : list of string
        The codes names you want to study.

    x_range : list of float, length 2
        The min and max magnitudes you want to do your study on.

    nb_bins : int
        The number of bins to study.

    Return
    ----------
    Nothing

    """

    description = st.expander("More information about this figure.")

    description.markdown(
        "Uncertainty calibration for the selected dataset. \
        The x-axis represents bins of VIS like magnitude. The y- \
        axis presents the fraction of object per bin of magnitude for \
        which the True value of a parameter falls in a an interval of the \
        predicted 1$\sigma$ uncertainty centered on the predicted value. \
        The final bin is for all objects, regardless of their magnitude."
    )
    x_min, x_max = x_range
    x_bins = np.linspace(x_min, x_max, nb_bins + 1)
    calib_mag = compute_error_prediction(df, params, codes, x_bins, nb_bins)
    fig = plot_error_prediction(dataset, calib_mag, params, codes, x_bins, LABELS)
    st.pyplot(fig)


def bt_multiband(
    df,
    bands,
    codes,
    outlier_threshold,
    abs=False,
):

    """
    Compute and plot the Summary Figure for b/t in the multi-band case, 
    representing the different metrics (Bias, Dispersion and Outlier fraction)
    of the b/t fitting for different codes and bands.

    Parameters
    ----------
    df : panda dataframe
        The catalogue of the dataset, with all codes and parameters

    bands : list of string
        The bands names you want to study.

    codes : list of string
        The codes names you want to study.

    outlier_threshold : float
        The threshold to define what is considered as an outlier

    abs : boolean, optional, default False
        If True, the absolute value of the bias is taken

    Raise
    ----------
    If not enough galaxies, raise a corresponding error messsage

    Return
    ----------
    results: dictionary
        The dictionary needed to do the plot

    """

    description = st.expander("More information about this figure.")
    description.markdown(
        """
        Summary plot for the mutli-band bulge-over-total ratio. 
        The different rows show the result for bright / intermediate / faint galaxies.
        The first column represents the mean bias $\mathcal{B}$ (see Eqs. 5 and 4 of the paper)
        regarding the different Euclid and Rubin bands, by increasing center wavelength.
        The second column, shows the dispersion $\mathcal{D}$ (see Eq. 6).
        The third column represents the fraction of outliers, $\mathcal{O}$.
        Each selected code is plotted with a different color as labelled.
        When all the available bands are selected, the different bands are shaded
        in red if they are Euclid bands, and in blue if Rubin.
        """
    )

    try:
        results = compute_bt_multiband(
            df,
            codes,
            bands,
            outlier_limit=outlier_threshold,
            abs=abs,
        )

        figure = bt_multiband_plot(
            results,
            labels=LABELS)
        st.pyplot(figure)

    except RuntimeError as e:
        st.markdown(f"## {e}")

    return results
