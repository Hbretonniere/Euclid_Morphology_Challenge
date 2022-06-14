from email.policy import default
import streamlit as st

import numpy as np
from app.io import load_data, save_results, load_data_photometry, import_TU_std
from app.help import readme
from app.params import (
    single_sersic_params,
    double_sersic_params,
    double_sersic_free_params,
    realistic_params,
    multiband_params,
    multiband_params_nbfix,
    LABELS,
)
from app.summary import summary, summary2D, trumpet, score, error_calibration, bt_multiband
from app.plots import photo_trumpet_plots, photo_trumpet_plots_multi_band
from app.utils import get_x_axis_range
import matplotlib.pyplot as plt

DATASETS = ("single_sersic", "double_sersic", "realistic", "multiband")
PARAMETERS_2D = ["re", "q"]

def photometry():

    description = st.expander("README")
    description.markdown(readme)

    demo = st.checkbox(
        "Demo version (much faster). Uncheck when all set to get the full results.",
        value=False,
    )
    st.title("Euclid Morphology Challenge DIY: Photometry 💡 ")

    
    st.sidebar.markdown("## Controls")
    st.sidebar.markdown(
        "Adjust the values below and the figures will be updated accordingly"
    )
    plot_type = st.sidebar.radio("Select a Type of plot", ["Summary Plots", "Trumpet Plots"], index=1)
    
    dataset = st.sidebar.radio(
        "Select a Dataset", DATASETS, format_func=lambda x: LABELS[x]
    )

    nb_free = None
    compo = None
    bands = None
    if dataset == "single_sersic":
        dataset_params = single_sersic_params
    elif dataset == "realistic":
        dataset_params = realistic_params
    elif dataset == "double_sersic":
        dataset_params = double_sersic_params
        nb_free = st.sidebar.checkbox("Use free bulge Sersic fit")
        if nb_free:
            dataset_params = double_sersic_free_params
        
    elif dataset == "multiband":
        dataset_params = multiband_params
        nb_free = st.sidebar.checkbox("Use free bulge Sersic fit")
        if not nb_free:
            dataset_params = multiband_params_nbfix
        bands = st.sidebar.multiselect(
                "Select Bands to display",
                dataset_params["bands_photo"],
                default=dataset_params["bands_photo"],
                # format_func=lambda x: LABELS[x],
            )

    # #####  Composant OPTIONS ####
    if dataset in ["double_sersic", "multiband"]:
        compo = st.sidebar.radio("Select the composante", ['total'])
        # compo = st.sidebar.radio("Select the composante", ['total', 'bulge', 'disk'])

    # #####  SOFTWARE OPTIONS ####
    all_code = st.sidebar.checkbox("Plot all software")
        
    if all_code:
        codes = dataset_params["available_codes"]
    else:
        codes = st.sidebar.multiselect(
            "Select software to display",
            dataset_params["available_codes"],
            default=dataset_params["available_codes"],
            format_func=lambda x: LABELS[x+'_'],
        )
        if len(codes) == 0:
            st.markdown("## Select at least one software to plot !")
            return 0
        
    # #####  Fields OPTIONS ####
    fields = None
    if dataset != 'multiband':
        all_fields = st.sidebar.checkbox("Plot all Fields")
        if all_fields:
            fields = ['1', '2', '3', '4']
        else:
            fields = st.sidebar.multiselect(
                "Select fields to display",
                ['1', '2', '3', '4'],
                default=['4'],
            )
            if len(fields) == 0:
                st.markdown("## Select at least one Field to plot !")
                return 0

    dfs = load_data_photometry(dataset, codes, nb_free, fields, demo, compo, bands)
    TU_stds = import_TU_std()
    if plot_type == 'Trumpet Plots':
        if dataset == 'multiband':
            st.write('multi')
            photo_trumpet_plots_multi_band(dfs, codes, bands, LABELS, TU_stds, compo, nb_free)
        else:
            photo_trumpet_plots(dfs, codes, fields, LABELS, TU_stds, compo, nb_free)
    elif plot_type == 'Summary Plots':
        st.markdown("### Not implemented yet")
        return 0

def morphology():
    
    description = st.expander("README")
    description.markdown(readme)

    demo = st.checkbox(
        "Demo version (much faster). Uncheck when all set to get the full results.",
        value=True,
    )
    st.title("Euclid Morphology Challenge DIY: Morphology *⬬*")

    nb_free = False
    band = None

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
            format_func=lambda x: LABELS[x+'_'],
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
    if dataset == 'multiband':
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
            y_lims = st.slider(
                "x-axis range", min_value=-5., max_value=10., value=[-1., 1.], step=0.1
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
        elif plot_type == "Trumpet Plots":
            results = trumpet(df, params, codes, x_axis, xs , n_bins, outliers, y_lims)
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



page_names_to_funcs = {
    "⬬ Morphology (EMC Part II)": morphology,
    "💡 Photometry (EMC Part I)": photometry,
}


selected_page = st.sidebar.selectbox("Choose the EMC paper", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
