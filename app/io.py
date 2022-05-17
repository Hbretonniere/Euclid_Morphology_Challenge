import os
import pickle
import numpy as np
from astropy.table import Table
import streamlit as st
import pandas as pd

RESULTS_DIR = "results"


def read_catalogue(filename):
    """
    Transform an astropy Table to a panda dat

    Parameters
    ----------
    filename : string
        The name of the catalogue file

    Return
    ----------
    The dataframe containing all the galaxies / codes / fitted parameters

    """
    return Table.read(filename).to_pandas()


@st.cache(suppress_st_warning=True)
def load_data(dataset, band=None, nb_free=False, demo=False):
    """
    Load the wanted catalogues. All the catalogues have the same name structures,
    which can be loaded thanks to the values returned by the different buttons selected in the
    main.

    Parameters
    ----------
    dataset : string
        The name of the dataset you want to study, e.g. 'single_sersic'.

    band : boolean, optional, default False
        If True, load for the wanted band among the multiband catalogues

    nb_free : boolean, optional, default False
        If True, load the double sersic catalogue fitted with a free bulge Sersic index model.
        If False, load the fixed bulge Sersic fit model.

    demo : boolean, default False
        If True, load only 1/100 of the catalogue. Used for exploring the app
        Must be False to have final scientific results

    Return
    ----------
    The dataframe containing all the galaxies / codes / fitted parameters

    """

    nb_free_prefix = "nb_free_" if nb_free else ""
    band = "" if band is None else f"_{band}"

    filename = f"data/challenge_5sig_overlap_{nb_free_prefix}{dataset}{band}.fits"
    # filename = f"data/challenge_5sig_overlap_single_sersic_wometrykastars.fits"
    # filename = f"data/gala_constrained_challenge_5sig_overlap_double_sersic.fits"

    if not os.path.exists(filename):
        st.markdown('# Downloading the catalogues, can take some time...')
        os.system('zenodo_get -o ./data/ 10.5281/zenodo.6421906')
    
    cat = read_catalogue(filename)


    if demo:
        # Use only 1% of the full catalogue
        return cat[::100]

    return cat


def save_results(results, dataset, nb_free):
    """
    Save the metrics results dictionary in a pickle file


    Parameters
    ----------
    results : list of dictionary, length 3
        results = [bias, dispersion, outlier fraction]
        each metric is a dictionary, containing the result of the metric for each code and parameters plotted

    dataset : string
        The name of the selected dataset. Used for customizing the name of the saved file.

    nb_free : string
        "True" or "False. Used for customizing the name of the saved file if the free bulge sersic model fit
        option is selected.

    Return
    ----------
    filepath: string
        The name of the file path which will be written below the plot.

    """

    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"MorphoChallenge_results_{dataset}_nbfree_{nb_free}.pickle"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return filepath


def load_data_photometry(dataset, codes, nb_free, fields, demo, bands=None):
    if dataset == 'single_sersic':
        nb_free_prefix = ""
    else:
        nb_free_prefix = "_bf" if nb_free else "_nb4"
    cats = {}
    for code in codes:
        for field in fields:
            band = "" if bands is None else f"_{band}"

            filename = f"data/plots_photometry/{code}_{dataset}{nb_free_prefix}/{code}_{dataset}{nb_free_prefix}_{field}.dat"
            # print(filename)

            # if not os.path.exists(filename):
            #     st.markdown('# Downloading the catalogues, can take some time...')
            #     os.system('zenodo_get -o ./data/ 10.5281/zenodo.6421906')
            
            # cat = pd.read_fwf(filename)
            cat = np.loadtxt(filename)
            if demo:
                cat = cat[::100]
                
            cats[f'{code}_{field}'] = cat

    return cats