import os
import pickle
import numpy as np
from astropy.table import Table
import streamlit as st
import pandas as pd
import h5py
from astropy.io import ascii as astro_ascii

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
        os.system('zenodo_get -o ./data/ 10.5281/zenodo.6421905')
    
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


def load_data_photometry(dataset, codes, nb_free, fields, demo, composant=None, bands=None):
    
    try:
        hf = h5py.File('data/EMC_photometry.hdf5', 'r')
    except:
        st.markdown('# Downloading the catalogues, can take some time...')
        os.system('zenodo_get -o ./data/ 10.5281/zenodo.6421905')
        hf = h5py.File('data/EMC_photometry.hdf5', 'r')
        
    if dataset in ['single_sersic', 'realistic']:
        nb_free_prefix = ""
        composant_prefix = ""
    else:
        nb_free_prefix = "_bf" if nb_free else "_b4"
        composant_prefix = f'_{composant}'
    cats = {}
    if dataset == 'multiband':
        fields = [0]
    for code in codes:
        # if code == 'deepleg':
            # nb_free_prefix = ""
        if dataset == 'double_sersic':
            nb_free_prefix = "_bf" if nb_free else "_b4"

        for field in fields:
            if bands == None:
                if code == 'deepleg':
                    name = f"{code}_{dataset}/{code}_{dataset}_{field}{nb_free_prefix}{composant_prefix}"
                else:
                    name = f"{code}_{dataset}{nb_free_prefix}/{code}_{dataset}_{field}{nb_free_prefix}{composant_prefix}"
                cat = hf[name][()]
                if demo:
                    cat = cat[::100]
                
                cats[f'{code}_{field}'] = cat
            else:
                for band in bands:
                    name = f"{code}_{dataset}/{code}_{dataset}_{band}{nb_free_prefix}{composant_prefix}"
                    # st.write(name)
                    cat = hf[name][()]
                    if demo:
                        cat = cat[::100]
                
                    cats[f'{code}_{band}'] = cat

            

    return cats

def import_TU_std():
    
    e = astro_ascii.read('./data/TU_std_15bins14-28_bands.dat')

    TUstds={} 
    TUstds['lsst_u']=np.lib.recfunctions.structured_to_unstructured(np.array(e[1]))
    TUstds['lsst_g']=np.lib.recfunctions.structured_to_unstructured(np.array(e[2]))
    TUstds['lsst_r']=np.lib.recfunctions.structured_to_unstructured(np.array(e[3]))
    TUstds['lsst_i']=np.lib.recfunctions.structured_to_unstructured(np.array(e[4]))
    TUstds['lsst_z']=np.lib.recfunctions.structured_to_unstructured(np.array(e[5]))
    TUstds['vis']=np.lib.recfunctions.structured_to_unstructured(np.array(e[6]))
    TUstds['nir_y']=np.lib.recfunctions.structured_to_unstructured(np.array(e[7]))
    TUstds['nir_j']=np.lib.recfunctions.structured_to_unstructured(np.array(e[8]))
    TUstds['nir_h']=np.lib.recfunctions.structured_to_unstructured(np.array(e[9]))
    return TUstds