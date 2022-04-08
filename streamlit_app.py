import streamlit as st

import numpy as np
from app.io import load_data, save_results
from app.help import readme
from app.params import (
    single_sersic_params,
    double_sersic_params,
    double_sersic_free_params,
    realistic_params,
    multiband_params,
    LABELS,
)
from app.summary import summary, summary2D, trumpet, score, error_calibration, bt_multiband
from app.utils import get_x_axis_range

DATASETS = ("single_sersic", "double_sersic", "realistic", "multiband")
PARAMETERS_2D = ["re", "q"]


def main():
    """
    Create the different buttons and setting of the app. The description of the buttons are in help.py
    The buttons and setting depend of the selected dataset. The differences are written in the dictionaries
    defined in params.py

    Load the wanted dataset, defined in io.py

    Call the different actions defined in summary.py,
    which call someroutines of utils.py and the plotting routines defined in plot.py

    Launch the web page with the interface
    """
    nb_free = False
    band = None

    description = st.expander("README")
    description.markdown(readme)

    st.title("MorphoChallenge DIY plots")
    demo = st.checkbox(
        "Demo version (much faster). Uncheck when all set to get the full results.",
        value=True,
    )

    st.sidebar.markdown("## Controls")
    st.sidebar.markdown(
        "Adjust the values below and the figures will be updated accordingly"
    )

    dataset = st.sidebar.radio(
        "Select a Dataset", DATASETS, format_func=lambda x: LABELS[x]
    )
    if dataset == "single_sersic":
        dataset_params = single_sersic_params
    elif dataset == "realistic":
        dataset_params = realistic_params
    elif dataset == "double_sersic":
        dataset_params = double_sersic_params
        nb_free = st.sidebar.checkbox("Use free bulge Sersic fit")
    elif dataset == "multiband":
        dataset_params = multiband_params
        # band = st.sidebar.radio("Which fitted band ?", ["VIS", "NIR-y"])

        # if band == "NIR-y" and "SE++" in dataset_params["available_codes"]:
            # dataset_params["available_codes"].remove("SE++")
        # results = bt_multiband(df, bands, codes, outliers, 'Truemag', factors)

    if nb_free:
        dataset_params = double_sersic_free_params

    df = load_data(dataset, nb_free=nb_free, band=band, demo=demo)

    plot_type = st.sidebar.radio("Select a Type of plot", dataset_params["plot_types"])

    #  ### PARAMETERS OPTIONS ####
    if dataset != 'multiband':
        all_params = st.sidebar.checkbox("Plot all parameters")
        if all_params:
            params = dataset_params["available_params"][plot_type]
        else:
            params = st.sidebar.multiselect(
                "Select relevant parameters",
                dataset_params["available_params"][plot_type],
                default=[
                    dataset_params["available_params"][plot_type][0],
                    dataset_params["available_params"][plot_type][1],
                ],
                format_func=lambda x: LABELS[x],
            )
            if len(params) == 0:
                st.markdown("## Choose at least one parameter to plot !")
                return 0

    # #####  SOFTWARE OPTIONS ####
    all_code = st.sidebar.checkbox("Plot all software")
    if all_code:
        codes = dataset_params["available_codes"]
    else:
        codes = st.sidebar.multiselect(
            "Select software to display",
            dataset_params["available_codes"],
            default=dataset_params["available_codes"],
            format_func=lambda x: LABELS[x],
        )
        if len(codes) == 0:
            st.markdown("## Select at least one software to plot !")
            return 0

    # #### OUTLIERS OPTIONS ####
    outliers = st.slider(
        "Outliers Threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.05
    )

    #  #### X AXIS OPTIONS ####
    x_min, x_max = 28, 25.3
    if dataset == 'mutliband':
        all_bands = st.sidebar.checkbox("Plot all bands")
        if all_bands:
            bands = dataset_params['bands']
        else:
            bands = st.sidebar.multiselect(
                "Select software to display",
                dataset_params["bands"],
                default=dataset_params["bands"],
                # format_func=lambda x: LABELS[x],
            )
            if len(bands) == 0:
                st.markdown("## Select at least one band to plot !")
                return 0
    elif plot_type == "2D Summary Plots":
        default_bins = 5
        x_axis = "Truemag"
        mag_min, mag_max = st.slider(
            "VIS True magnitude range",
            min_value=18.0,
            max_value=26.0,
            value=[19., 26.],
            step=0.05,
        )
        mags = [mag_min, mag_max]
        bt_min, bt_max = st.slider(
            "bt range", min_value=0.0, max_value=1.0, value=[0.1, 0.9], step=0.1
        )
        bts = [bt_min, bt_max]
        n_bins_mag = st.sidebar.number_input(
            "Number of mag bins", min_value=5, max_value=10, value=default_bins
        )
        n_bins_bt = st.sidebar.number_input(
            "Number of bt bins", min_value=5, max_value=10, value=default_bins
        )

    else:
        x_axis = st.sidebar.radio("X axis", dataset_params["x_axis_options"])
        default_bins = 11
        if plot_type == "Trumpet Plots":
            y_max = st.slider(
                "Bias range", min_value=0.3, max_value=5.0, value=1.0, step=0.1
            )
       
        xs = get_x_axis_range(dataset, x_axis, demo)
        n_bins = st.sidebar.number_input(
            f"Number of {LABELS[x_axis]} bins",
            min_value=2,
            max_value=15,
            value=dataset_params["default bins"][plot_type],
        )

    # #### More options ####
    more_options = st.sidebar.checkbox("More options")
    absolute = False
    factors = [1, 1, 1, 1]
    show_scores = False
    bd_together = False
    if more_options:
        cut_outliers = st.sidebar.checkbox("Do not remove outliers")
        if plot_type == "Summary Plots":
            show_scores = st.sidebar.checkbox("Show_scores")
        if cut_outliers:
            outliers = 1000
        absolute = st.sidebar.checkbox("Absolute error")

    # #### CREATE AND PLOT ####
    
    if dataset == 'multiband':
        results = bt_multiband(df, bands, codes, outliers)
    else:

        if plot_type == "Summary Plots":
            limit_y_axis = st.checkbox('Limit y axis range', value=True)
            results = summary(
                df,
                dataset,
                params,
                codes,
                xs,
                n_bins,
                outliers,
                x_axis,
                score_factors=factors,
                limit_y_axis=limit_y_axis,
                abs=absolute,
                show_scores=show_scores,
                bd_together=bd_together,
            )
            # st.write(results)
        elif plot_type == "Trumpet Plots":
            results = trumpet(df, params, codes, x_axis, xs , n_bins, outliers, y_max)
        elif plot_type == "2D Summary Plots":
            results = summary2D(df, params, codes, mags, n_bins_mag, bts, n_bins_bt, outliers)
        elif plot_type == "Summary Scores":
            results = score(
                df, dataset, params, codes, factors, xs, n_bins, outliers, x_axis, abs=False
            )
        elif plot_type == "Error Prediction":
            results = error_calibration(df, dataset, params, codes, xs, n_bins)

        if st.button("Save results (dictionnary)", disabled="Summary" not in plot_type):
            filepath = save_results(results, dataset, nb_free)
            st.success(f"Results saved as {filepath}")


if __name__ == "__main__":
    main()
