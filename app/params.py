single_sersic_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "n"],
        "Summary Scores": ["re", "q", "n"],
        "Trumpet Plots": ["re", "q", "n"],
        "Error Prediction": ["re", "q", "n"],
    },
    "available_codes": ["deepleg", "gala", "metryka", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 11,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
    "x_axis_options": ['Truemag', 'Truere', 'Truez', 'Truen']
}

double_sersic_params = {
    "available_params": {
        "Summary Plots": ["bt", "reb", "red", "qb", "qd"],
        "2D Summary Plots": ["re", "q"],
        "Summary Scores": ["bt", "re", "q"],
        "Trumpet Plots": ["reb", "red", "qb", "qd", "bt"],
        "Error Prediction": ["reb", "red", "qb", "qd"],
    },
    "available_codes": ["deepleg", "gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "2D Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 11,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
    "x_axis_options": ['Truemag', 'Truered', 'Truebt']
}

multiband_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "bt"],
        "2D Summary Plots": ["re", "q"],
        "Summary Scores": ["re", "q", "bt"],
        "Trumpet Plots": ["reb", "red", "qb", "qd", "bt"],
        "Error Prediction": ["reb", "red", "qb", "qd"],
    },
    "available_codes": ["gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots"
    ],
    "default bins": {
        "Summary Plots": 11,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
    "bands":['u_lsst', 'g_lsst', 'r_lsst', 'i_lsst', 'vis', 'z_lsst', 'y_nir', 'j_nir', 'h_nir'],
    "bands_photo":['lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'vis', 'lsst_z', 'nir_y', 'nir_j', 'nir_h']
    }

multiband_params_nbfix = {
    "available_params": {
        "Summary Plots": ["re", "q", "bt"],
        "2D Summary Plots": ["re", "q"],
        "Summary Scores": ["re", "q", "bt"],
        "Trumpet Plots": ["reb", "red", "qb", "qd", "bt"],
        "Error Prediction": ["reb", "red", "qb", "qd"],
    },
    "available_codes": ["gala", "profit"],
    "plot_types": [
        "Summary Plots"
    ],
    "default bins": {
        "Summary Plots": 11,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
    "bands":['u_lsst', 'g_lsst', 'r_lsst', 'i_lsst', 'vis', 'z_lsst', 'y_nir', 'j_nir', 'h_nir'],
    "bands_photo":['lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'vis', 'lsst_z', 'nir_y', 'nir_j', 'nir_h']
    }


realistic_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "n"],
        "Summary Scores": ["re", "q", "n"],
        "Trumpet Plots": ["re", "q", "n"],
        "Error Prediction": ["re", "q", "n"],
    },
    "available_codes": ["gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 11,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
    "x_axis_options": ['Truemag', 'Truere', 'Truez', 'Truen']
}


double_sersic_free_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "BulgeSersic", "bt"],
        "2D Summary Plots": ["re", "q"],
        "Summary Scores": ["re", "q", "bt"],
        "Trumpet Plots": ["reb", "red", "qb", "qd", "bt"],
        "Error Prediction": ["reb", "red", "qb", "qd"],
    },
    "available_codes": ["gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "2D Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 11,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
    "x_axis_options": ['Truemag', 'Truere', 'Truez']
}
LABELS = {
    "single_sersic": "Single Sersic",
    "double_sersic": "Double Sersic",
    "realistic": "Realistic",
    "multiband": "Multi-band",
    "mag": "Magnitude",
    "re": "Effective radius",
    "reb": "Bulge radius",
    "red": "Disk radius",
    "q": "Axis ratio",
    "qb": "Bulge axis ratio",
    "qd": "Disk axis ratio",
    "n": "Sérsic index",
    "BulgeSersic": "Bulge Sersic index",
    "bt": "Bulge-to-total \n flux ratio",
    "SE++_": 'SourceXtractor++',
    "gala_": 'Galapagos-2',
    "deepleg_": 'DeepLeGATo',
    "profit_": 'ProFit',
    "metryka_": 'Morfometryka',
    "SE++": 'SourceXtractor++',
    "gala": 'Galapagos-2',
    "deepleg": 'DeepLeGATo',
    "profit": 'ProFit',
    "metryka": 'Morfometryka',
    "True Magnitude": "$I_{\mathrm{\mathsf{E}}}$ true magnitude",
    "True radius": "True radius",
    "True redshift": "True redshift",
    "Truere": "True  radius",
    "Truered": "True disk radius",
    "True disk radius": "True disk radius",
    "Truemag": "$I_{\mathrm{\mathsf{E}}}$ true magnitude",
    "Truez": "True redshift",
    'Truen': "True Sérsic index",
    "True b/t": "$I_{\mathrm{\mathsf{E}}}$ true b/t",
    "Truebt": "$I_{\mathrm{\mathsf{E}}}$ true b/t",
    "mu": r"$\mu$",
    "Bulge re": "Bulge \n radius",
    "Disk re": "Disk \n radius",
    "Bulge q": "Bulge \n axis ratio",
    "Disk q": "Disk \n axis ratio",
    "Bulge mu": r"Bulge $\mu$",
    "Disk mu": r"Disk $\mu$",
    'vis': r'$I_{\scriptscriptstyle\rm E}$',
    'j_nir': r'$J_{\scriptscriptstyle\rm E}$',
    'y_nir': r'$Y_{\scriptscriptstyle\rm E}$',
    'h_nir': r'$H_{\scriptscriptstyle\rm E}$',
    'g_lsst':r'\emph{g}',
    'r_lsst':r'\emph{r}',
    'i_lsst':r'\emph{i}',
    'u_lsst':r'\emph{u}',
    'z_lsst':r'\emph{z}'
}

y_lims = {'re':{'B':[-0.1, 0.12], 'D':[-0.05, 0.25], 'O':[-0.05, 0.32]},
            'q':{'B':[-0.07, 0.02], 'D':[-0.02, 0.21], 'O':[-0.001, 0.013]},
            'n':{'B':[-0.08, 0.1], 'D':[-0.03, 0.3], 'O':[-0.01, 0.155]},
            'reb':{'B':[-0.6, 45.], 'D':[-0.4, 62.], 'O':[-0.05, 1.]},
            'red':{'B':[-0.05, 0.2], 'D':[-0.01, 0.5], 'O':[-0.02, 0.45]},
            'qb':{'B':[-0.5, 0.3], 'D':[-0.05, 1.2], 'O':[-0.05, 0.48]},
            'qd':{'B':[-0.25, 0.1], 'D':[-0.05, 0.55], 'O':[-0.03, 0.16]},
            'bt':{'B':[-0.3, 0.4], 'D':[-0.01, 0.25], 'O':[-0.03, 0.31]},
            'BulgeSersic':{'B':[-0.28, 0.1], 'D':[-0.02, 0.63], 'O':[-0.03, 0.20]}
}