import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams as mp_param
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import streamlit as st
from app.params import y_lims
from app.utils import define_y_axis_slider, chose_mode, compute_summary, sub_cat, find_outliers, compute_error_bin

plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
# plt.rcParams['text.usetex'] = True

# @st.cache
def trumpet_plot(
    cat,
    codes,
    param,
    outlier_threshold,
    x_axis,
    x_range,
    y_range,
    nb_bins,
    labels,
    freq_scat=1,
    abs=False,
):

    """
    Compute and create the Scatter plot figure for the Bias of a galaxy parameter regarding its magnitude,
    for different codes.
    In the plot, the running mean and std of each bin of magnitude is plotted in solid orange.
    Markers transparency are proportional to the density of points.
    Also plot the bias distribution in the right of the main plot.

    Parameters
    ----------
    cat : pandas dataframe
        The input catalogue to compute the biases.

    codes : list of strings
        The list of codes names you want to plot. Each code will have a different subplot.

    param : string
        The name of the parameter you want to plot. It needs to be one key of the catalogue cat.

    outlier_threshold : float
        The threshold to define what is considered as an outlier. Dashed red lines will be plotted at this
        value ( + threshold and - threshold). Points above and below the threshold are removed to plot
        the running mean and bias (orange lines)

    x_range : list of float, of len two
        The x-range for each subplot.

    y_range : list of float, of len two
        The y-range for each subplot.

    nb_bins : int
        The number of bin to compute the running mean and std.

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    freq_scat : int, optional, default 10
        The fraction of object to be plotted in the scatter plot

    x_axis : string, optional, default "True magnitude"
        Define the parameter to plot the Bias against. By default, the magnitude.

    abs : boolean, optional, default: False
        If True, the absolute value of the Bias is plotted.

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    if len(codes) > 4:
        nb_columns = 7
        dim = (15, 17)
        legend_y = -3.8
    elif len(codes) > 2:
        nb_columns = 7
        dim = (15, 15)
        legend_y = -2.1
    else:
        nb_columns = 3
        dim = (10, 10)
        legend_y = -0.25

    fig = plt.figure(constrained_layout=True, figsize=dim)
    
    if len(codes) < 2:
        y_suptitle = 1
    elif param == 'bt':
        y_suptitle = 0.98
    else:
        y_suptitle = 0.95

    fig.suptitle(labels[param], y=y_suptitle, fontsize=35)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.6
    )

    nb_lines = int(np.ceil(len(codes) / 2))

    gs = GridSpec(nb_lines, nb_columns, figure=fig)

    if len(cat[f"True{param}"]) > 3e3:
        alpha = 0.01
    else:
        alpha = 0.8
        #alpha = 0.1

    line = 0
    mode = chose_mode(param)

    cmaps = {
        "SE++": plt.cm.Greys,
        "metryka": plt.cm.Reds,
        "profit": plt.cm.Blues,
        "gala": plt.cm.Greens,
        "deepleg": plt.cm.Purples,
    }
    for i, code in enumerate(codes):
        cmaps[code] = LinearSegmentedColormap.from_list(
            "mycmap", cmaps[code](np.linspace(0.4, 1, 1000))
        )
        if i % 2 == 0:
            grid_scat = fig.add_subplot(gs[line, :2])
            grid_hist = fig.add_subplot(gs[line, 2:3])
        else:
            grid_scat = fig.add_subplot(gs[line, 4:6])
            grid_hist = fig.add_subplot(gs[line, 6:7])
            line += 1
        grid_hist.set_xticks([])
        grid_scat.set_title(labels[code], fontsize=25, usetex=False)

        code_cat, _ = sub_cat(cat, x_axis, x_range, 0)
        X_true = code_cat[x_axis]
        error = compute_error_bin(code_cat, code, param, mode, abs=abs)
        if len(np.argwhere(np.isnan(error.values))) > 0:
            na = np.argwhere(np.isnan(error.values))
            error.drop(error.index[na[0]], inplace=True)
            X_true.drop(X_true.index[na[0]], inplace=True)
        indices_out = find_outliers(code_cat, code, param, mode, y_range[1], abs=abs)
        error_dropped = error.drop(error.index[indices_out])
        X_true_dropped = X_true.drop(X_true.index[indices_out])

        good_mean = np.median(error)
        bins = [200, 200]
        histo, locx, locy = np.histogram2d(
            X_true_dropped[::freq_scat], error_dropped[::freq_scat], bins=bins
        )
        density = np.array(
            [
                histo[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])]
                for a, b in zip(X_true_dropped, error_dropped)
            ]
        )
        sorted_indices = density.argsort()[::-1]
        color = density[sorted_indices][::freq_scat]

        grid_scat.scatter(
            X_true_dropped.iloc[sorted_indices][::freq_scat],
            error_dropped.iloc[sorted_indices][::freq_scat],
            c=color,
            cmap=cmaps[code],
            marker=".",
            alpha=alpha,
            rasterized=True,
        )

        grid_scat.axhline(
            0.0, color="w", linestyle="-", linewidth=1.8
        )
        grid_scat.axhline(
            0.0, color="#0000CD", linestyle="-", linewidth=1.3
        )
        grid_hist.set_xlabel('Density\ncount', fontsize=28, usetex=False)
        """ Plot the running mean  """
        x_min, x_max = x_range
        nb_bins = 6
        rot = 45

        if x_axis in ['Truere']:
            x_bins = np.logspace(np.log10(x_min), np.log10(x_max), nb_bins + 1)

        else:
            x_bins = np.linspace(x_min, x_max, nb_bins+1)

        for tick in grid_scat.get_xticklabels():
            tick.set_rotation(rot)

        means, stds, _, _, _ = compute_summary(cat, [param], [code], x_bins, outlier_threshold, x_axis=x_axis)
        mean = means[param][code]
        std = stds[param][code]

        mean_bins =  (x_bins[1:] + x_bins[:-1]) * 0.5

        grid_scat.errorbar(
            mean_bins,
            mean,
            yerr=std,
            color="darkorange",
            linewidth=3,
            elinewidth=4,
            capsize=5,
            markeredgewidth=3,
            label=r"running bias $\tilde{\mathcal{B}}$ with dispersion $\tilde{\mathcal{D}}$ error bars",
        )
        grid_scat.errorbar(
            mean_bins,
            mean,
            yerr=std,
            color="white",
            linewidth=0.8,
            elinewidth=1,
            capsize=3,
            markeredgewidth=0.8,
        )
        grid_scat.axhline(
            -outlier_threshold,
            ls="--",
            color="red",
            linewidth=0.8,
        )
        grid_scat.axhline(
            outlier_threshold,
            ls="--",
            color="red",
            linewidth=0.8,
        )

        
        grid_scat.set_ylim(y_range[0], y_range[1])
        grid_hist.set_ylim(y_range[0], y_range[1])
        grid_scat.set_yticks(grid_scat.get_yticks()[::2])
        grid_scat.set_yticklabels(grid_scat.get_yticks(), usetex=False)
        
        grid_scat.tick_params(axis="both", which="major", labelsize=25)
        grid_scat.set_xlabel(labels[x_axis], fontsize=24, usetex=False)

        grid_scat.set_xticks(x_bins[::2])
        grid_scat.set_xticklabels(np.round(grid_scat.get_xticks(), 1), usetex=False, fontsize=22)
        grid_scat.set_yticklabels(np.round(grid_scat.get_yticks(), 2), usetex=False, fontsize=22)

        if param == 'n':
            grid_scat.set_ylabel(
                r"$\mathrm{Pred}_{\log_{\mathrm{10}}({n})} - \mathrm{True}_{\log_{\mathrm{10}}({n})} $",
                fontsize=25, usetex=False
            )
        elif param == 'q':
            grid_scat.set_ylabel(
                r"$\mathrm{Pred}_{q} - \mathrm{True}_{q} $",
                fontsize=25, usetex=False
            )
        elif param == 're':
            grid_scat.set_ylabel(
                r'$\frac{\mathrm{Pred}_{\mathrm{r_{e}}} - \mathrm{True}_{r_{\mathrm{e}}}}{\mathrm{True}_{r_{\mathrm{e}}}} $',
                fontsize=35, usetex=False
            )
        elif param == 'red':
            grid_scat.set_ylabel(
                r'$\frac{\mathrm{Pred}_{r_{\mathrm{e, d}}} - \mathrm{True}_{r_{\mathrm{e, d}}}}{\mathrm{True}_{r_{\mathrm{e, d}}}} $',
                fontsize=35, usetex=False
            )
        elif param == 'reb':
            grid_scat.set_ylabel(
                r'$\frac{\mathrm{Pred}_{r_{\mathrm{e, b}}} - \mathrm{True}_{r_{\mathrm{e, b}}}}{\mathrm{True}_{r_{\mathrm{e, b}}}} $',
                fontsize=35, usetex=False
            )
        elif param == 'bt':
            grid_scat.set_ylabel(
                f"$\mathrm{{Pred_{{b/t}}}} - \mathrm{{True_{{b/t}}}}$",
                fontsize=30, usetex=False
            )
        
        elif param == 'qb':
            grid_scat.set_ylabel(
                r'$\mathrm{Pred}_{q_{\mathrm{b}}} - \mathrm{True}_{q_{\mathrm{b}}}$',
                fontsize=30, usetex=False
            )
        elif param == 'qd':
            grid_scat.set_ylabel(
                r'$\mathrm{Pred}_{q_{\mathrm{d}}} - \mathrm{True}_{q_{\mathrm{d}}}$',
                fontsize=30, usetex=False
            )
        
        else:
            grid_scat.set_ylabel(
                f"$\dfrac{{\mathrm{{Pred_{{{param}}}}} - \mathrm{{True_{{{param}}}}}}}{{\mathrm{{True_{{{param}}}}}}}$",
                fontsize=35, usetex=False
            )
        grid_scat.tick_params("both", length=3, width=1, which="minor")
        grid_scat.axhline(
            good_mean,
            ls="--",
            color="black",
            lw=2,
            alpha=1,
            label="Mean bias",
        )

        """ ############### Histogram ############### """
        cmaphist = plt.cm.get_cmap(cmaps[code])
        rgba = cmaphist(0.5)
        bins_hist = np.linspace(y_range[0], y_range[1], 500)
        histo = grid_hist.hist(
            error,
            bins=bins_hist,
            color=rgba,
            orientation="horizontal",
            lw=2,
            histtype="stepfilled",
        )
        grid_hist.set_yticks([])

        grid_hist.set_xlim([0, np.max(histo[0][3:-3])])
        grid_hist.hlines(
            good_mean,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            ls="--",
            color="black",
            lw=2,
            alpha=1,
        )
        grid_hist.hlines(
            0,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            color="#0000CD",
            lw=2,
            alpha=0.5,
        )
        grid_hist.hlines(
            -outlier_threshold,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            ls="--",
            color="red",
            linewidth=0.8,
        )
        grid_hist.hlines(
            outlier_threshold,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            ls="--",
            color="red",
            linewidth=0.8,
        )
        if x_axis == 'Truemag':
            grid_scat.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)

        if i == 0:
            grid_scat.legend(ncol=2, loc=(0, legend_y), fontsize=28)

    return fig


def plot_error_prediction(dataset, calib_mag, params, codes, x_bins, labels):

    """
    Create the error prediction calibration figure for different codes and parameters, regarding magnitude
    bins. The last bin is always the overall score (summary of all magnitude bins)

    Parameters
    ----------
    dataset : string
        The name of the dataset, e.g. 'single_sersic'. The number of subplots and place of the bars
        change accordingly

    calib_mag : dictionary of dictionaries
        Dictionary containing for each param the score of the different codes.
        e.g. calib_mag['re']['profit'] = value

    params : list of string
        The list of parameters name you want to plot. It needs to be one key of the dictionnary.

    codes : list of string
        The list of codes' name you want to plot. It needs to be one key of the dictionnary.

    x_bins : numpy array
        The bins of magnitude for which the error calibration have been computed on. Used for the x-label.

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    colors = {
        "SE++": "black",
        "gala": "#2F964D",
        "profit": "#2D7DBB",
        "metryka": "#D92523",
        "deepleg": "#7262AC",
    }

    if dataset in ["single_sersic", "realistic"]:
        pad = [-0.25, 0.25, -0.75, 0.75]
    else:
        pad = [-0.5, 0, 0.5]
    nb_lines = int(np.ceil(len(params) / 2))
    fig, ax = plt.subplots(nb_lines, 2, figsize=(10, nb_lines * 5))
    
    ax = ax.flatten()
    if len(params) % 2 != 0:
        ax[-1].set_visible(False)

    tick_labels = []
    for i in range(len(x_bins[:-1])):
        tick_labels.append(f"[{x_bins[i]:.1f}-{x_bins[i+1]:.1f}]")
    tick_labels.append("overall")

    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    bins = np.array(np.linspace(0, len(x_bins) * 3, len(x_bins)))

    for i, param in enumerate(params):
        j = 0
        for code in codes:
            if code == "deepleg":
                continue
            ax[i].bar(
                bins + pad[j],
                calib_mag[code][param],
                width=0.5,
                color=colors[code],
                label=labels[code],
            )
            j += 1
        ax[i].set_box_aspect(1)
        ax[i].set_xticks(bins)
        ax[i].set_xticklabels(tick_labels, fontsize=16, rotation=45, usetex=False)
        ax[i].set_xlabel("$I_{\mathrm{\mathsf{E}}}$ true magnitude", fontsize=20, usetex=False)
        ax[i].set_ylabel("Fraction of well \n calibrated objects", fontsize=19, usetex=False)
        ax[i].set_title(labels[param], fontsize=18, usetex=False)
        ax[i].axhline(0.68, ls="--", color="red")
        ax[i].text(0.1, 0.7, "0.68", size=20, usetex=False)
        ax[i].set_ylim([0, 0.9])
        
        fig.canvas.draw()
        
        ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize=16, usetex=False)
    ax[0].legend(fontsize=18, ncol=3, loc=(0, 1.2))
    [axe.axvline(13.3, ls='--', alpha=0.5) for axe in ax.flatten()]
    [axe.add_artist(Rectangle([13.2, -1], 3.5, 3.5, fill=True, alpha=0.08, color='blue')) for axe in ax.flatten()]
    return fig


def plot_score(scores, labels):

    """
    Plot the global score S for different catalogues and parameters.

    Parameters
    ----------
    scores : pandas dataframe
        The scores computed in compute_score. The columns represents the different codes,
        and the lines the parameters

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    sizes = {"SE++": 100, "profit": 150, "metryka": 100, "deepleg": 150, "gala": 100}

    colors = {
        "SE++": "black",
        "gala": "#2F964D",
        "profit": "#2D7DBB",
        "metryka": "#D92523",
        "deepleg": "#7262AC",
    }

    symbols = {"SE++": "o", "gala": "s", "profit": "*", "metryka": "D", "deepleg": "X"}

    fig, ax = plt.subplots(figsize=(12, 8))
    codes = scores.columns
    for code in codes:
        nb_param = len(scores[code].values)
        ax.scatter(
            np.arange(1, nb_param + 1),
            list(scores[code].values),
            label=labels[code],
            s=sizes[code],
            color=colors[code],
            marker=symbols[code],
            alpha=0.9,
        )

    ax.set_xticks(np.arange(1, nb_param + 1))
    ax.legend(fontsize=15)
    xlabels = []
    for param in scores.index.values:
        xlabels.append(labels[param])

    ax.set_xticklabels(xlabels, rotation=45, usetex=False)
    ax.set_ylabel(r"Global Score $\mathcal{S}$", fontsize=20, usetex=False)
    fig.canvas.draw()
    ax.set_yticklabels(ax.get_yticklabels(), usetex=False)
    if max(scores.max()) > 20:
        ax.set_yscale("log")
    return fig


def summary_plot(
    summary, x_bins, dataset, x_axis, labels, limit_y_axis, show_scores=False, bd_together=False
):
    """
    Create the Summary plot representing the different metrics (Bias, Dispersion and Outlier fraction)
    for different codes and parameters.

    Parameters
    ----------
    summary : List of dictionaries, of length 3
        The three metrics for each code and parameter, summary = [bias, dispersion, outlier_fraction]
        Each metric is a dictionary of dictionaries, e.g summary[0]['profit']['re'] = value
        The codes and parameters to plot will be automatically the one present in the dictionaries

    x_bins : numpy array
        The bins of the x-axis in which the summary has been computed on.

    dataset : string
        The name of the dataset, e.g. 'single_sersic'. The number of subplots changes accordingly

    x_axis : string
        Define the parameter to plot the metrics against. e.g. "True magnitude"

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    show_scores : boolean, optional, default False
        If True, print the global score S for each parameter and code on the right of the plot, ordered per
        ranking.

    bd_together ; boolean, optional, default False
        If True, plot the disk and bulges component together in the same subplot

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    colors = {
        "SE++": "black",
        "gala": "#2F964D",
        "profit": "#2D7DBB",
        "metryka": "#D92523",
        "deepleg": "#7262AC",
    }

    symbols = {"SE++": "o", "gala": "s", "profit": "*", "metryka": "D", "deepleg": "X"}

    mp_param.update({"font.size": 18})
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.tick_params(axis='x', which='minor')
    
    means = summary[0]
    stds = summary[1]
    outs = summary[2]
    params = list(means.keys())
    codes = list(means[list(means.keys())[0]].keys())
    c_legend = 'black'
    nb_params = max(2, len(params))
    x_ticks_bins = (x_bins[1:] + x_bins[:-1]) * 0.5


    nb_cols = 3
    if show_scores:
        nb_cols += 1

    fig, ax = plt.subplots(
        nb_params, nb_cols, figsize=(nb_cols * 5, (nb_params - 0.2) * 5)
    )
    plt.subplots_adjust(wspace=0.35, hspace=0.3)
    [axe.yaxis.set_major_formatter(FormatStrFormatter("%.2f")) for axe in ax.flatten()]
    [axe.xaxis.set_major_formatter(FormatStrFormatter("%.1f")) for axe in ax.flatten()]
    [
        axe.axhline(
            0, color="red", lw=5, alpha=0.5
        )
        for axe in ax[:, 0].flatten()
    ]
    # print(x_axis)
    if x_axis in ["Truemag", "Truez"]:

        rot = 45
        [
            axe.set_xlim(x_bins[0] - 0.5, x_bins[-1] + 0.5)
            for axe in ax.flatten()
        ]
        if x_bins[-1] == 25.3:
            max_ = 26
        else:
            max_ = x_bins[-1]
        [
            axe.set_xticks(np.round(np.linspace(x_bins[0], max_, 5), 1))
            for axe in ax.flatten()
        ]
        [
            axe.set_xticklabels(np.round(np.linspace(x_bins[0], max_, 5), 1), fontsize=22, usetex=False)
            for axe in ax.flatten()
        ]
    else:
        rot = 45
        [
            axe.set_xlim(x_bins[0] - 0.1, x_bins[-1] + 0.1)
            for axe in ax.flatten()
        ]
        [
            axe.set_xticks(np.linspace(x_bins[0], x_bins[-1], 5))
            for axe in ax.flatten()
        ]
        [
            axe.set_xticklabels(np.round(np.logspace(np.log10(x_bins[0]), np.log10(x_bins[-1]), 5), 2), fontsize=22, usetex=False)
            for axe in ax.flatten()
        ]
    legend_columns = 3
    fs = 25
    [ax[-1, i].set_xlabel(labels[x_axis], fontsize=fs, usetex=False) for i in [0, 1, 2]]
    ax[0, 0].set_title(r"Bias $\mathcal{B}$", fontsize=fs, usetex=False)
    ax[0, 1].set_title(r"Dispersion $\mathcal{D}$", fontsize=fs, usetex=False)
    ax[0, 2].set_title("Outlier Fraction $\mathcal{O}$", fontsize=fs, usetex=False)
    p = 0
    alpha = 0.55
    for p, param in enumerate(params):
        if ((param == 'q')):
            ax[p, -1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        for code in codes:
            # if (
                # (dataset == "realistic") | (dataset == "single_sersic") | (param in ["mag", "n", "bt", "BulgeSersic"])
            # ):
                if param == "reb":
                    ax[p, 0].set_yscale("symlog")
                    ax[p, 1].set_yscale("symlog")

                ax[p, 0].plot(
                    x_ticks_bins,
                    means[param][code],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                    alpha=alpha
                )

                ax[p, 1].plot(
                    x_ticks_bins,
                    stds[param][code],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                    alpha=alpha
                )
                ax[p, 2].plot(
                    x_ticks_bins,
                    outs[param][code],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                    alpha=alpha
                )

                if limit_y_axis:
                    ax[p, 0].set_ylim(y_lims[param]['B'])
                    ax[p, 1].set_ylim(y_lims[param]['D'])
                    ax[p, 2].set_ylim(y_lims[param]['O'])

                ax[0, 0].legend(fontsize=fs+0.1, ncol=legend_columns, loc=(0, 1.2))
                
                fig.canvas.draw()
                axy_labels = [item.get_text() for item in ax[p, 0].get_yticklabels()]
                ax[p, 0].set_yticklabels(axy_labels, usetex=False, fontsize=22)
                axy_labels = [item.get_text() for item in ax[p, 1].get_yticklabels()]
                ax[p, 1].set_yticklabels(axy_labels, usetex=False, fontsize=22)
                axy_labels = [item.get_text() for item in ax[p, 2].get_yticklabels()]
                ax[p, 2].set_yticklabels(axy_labels, usetex=False, fontsize=22)


                if param == 'n':
                    label = f'$\log_{{{10}}}$ Sérsic index'
                elif param == 'BulgeSersic':
                    label = f'$\log_{{{10}}}$ Bulge  \n Sérsic index'
                else:
                    label = labels[param]
                ax[p, 0].set_ylabel(label, color=c_legend, fontsize=fs, usetex=False)

    if len(params) == 1:

        if limit_y_axis:
            col1, col2, col3 = st.columns(3)
            min_b, max_b = define_y_axis_slider(means, param)
            min_d, max_d = define_y_axis_slider(stds, param)
            min_o, max_o = define_y_axis_slider(outs, param)
        
            with col1:
                b_min, b_max = st.slider('Bias range', min_b-min_b/10., max_b+max_b/10., value=[y_lims[param]['B'][-1], y_lims[param]['B'][0]], step=float(max(abs(min_b), abs(max_b))/10))

            with col2:
                d_min, d_max = st.slider('Dispersion range range', min_d-min_d/10., max_d+max_d/10., value=[y_lims[param]['D'][-1], y_lims[param]['D'][0]], step=float(max(abs(min_d), abs(max_d))/10))

            with col3:
                o_min, o_max = st.slider('Outlier fraction range', min_o-max_o/10., max_o+max_o/10., value=[y_lims[param]['O'][-1], y_lims[param]['O'][0]], step=float(max(abs(min_o), abs(max_o))/10))

            ax[0, 0].set_ylim([b_min, b_max])
            ax[0, 1].set_ylim([d_min, d_max])
            ax[0, 2].set_ylim([o_min, o_max])
        ax[-1, 0].set_visible(False)
        ax[-1, 1].set_visible(False)
        ax[-1, 2].set_visible(False)
        [ax[0, 0].set_xlabel(labels[x_axis], fontsize=fs, usetex=False) for i in [0, 1, 2]]

    # else:
        # [axe.set_xticklabels([]) for axe in ax[:-1, :].flatten()]

    for axe in ax.flatten():
        for tick in axe.get_xticklabels():
            tick.set_rotation(rot)
    # ax[1, 0].set_yticklabels([r'$0$', r'$10^0$', r'$10^1$'], fontsize=fs-5, usetex=False)
    # ax[1, 1].set_yticklabels([r'$0$', r'$10^0$', r'$10^1$'], fontsize=fs-5, usetex=False)

    return fig


def bt_multiband_plot(
    summary, labels
):
    """
    Create the bulge-over-total flux ratio for the multiband analysis. The plot
    represents the different metrics (Bias, Dispersion and Outlier fraction)
    for different codes, in three bins of magnitude, regarding different bands

    Parameters
    ----------
    summary : List of dictionaries, of length 3
        The three metrics for each code and parameter, summary = [bias, dispersion, outlier_fraction]
        Each metric is a dictionary of dictionaries, e.g summary[0]['profit']['bt'] = value
        The codes and bands to plot will be automatically the ones present in the dictionaries

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"bt": "Bulge-over-total flux ratio"}
    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    colors = {
        "SE++": "black",
        "gala": "#2F964D",
        "profit": "#2D7DBB",
        "metryka": "#D92523",
        "deepleg": "#7262AC",
    }

    symbols = {"SE++": "o", "gala": "s", "profit": "*", "metryka": "D", "deepleg": "X"}

    mp_param.update({"font.size": 18})
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    means = summary[0]
    stds = summary[1]
    outs = summary[2]
    codes = list(means.keys())
    bands = list(means[list(means.keys())[0]].keys())
    xs = np.arange(len(bands))
    nb_cols = 3

    fig, ax = plt.subplots(
        3, nb_cols, figsize=(nb_cols * 5, (3 - 0.2) * 5)
    )
    plt.subplots_adjust(wspace=0.35, hspace=0.3)
    [axe.yaxis.set_major_formatter(FormatStrFormatter("%.2f")) for axe in ax.flatten()]

    legend_columns = 3

    fs = 20
    [ax[-1, i].set_xlabel('Bands', fontsize=fs, usetex=False) for i in [0, 1, 2]]
    ax[0, 0].set_title(r"Bias $\mathcal{B}$", fontsize=fs, usetex=False)
    ax[0, 1].set_title(r"Dispersion $\mathcal{D}$", fontsize=fs, usetex=False)
    ax[0, 2].set_title("Outlier Fraction $\mathcal{O}$", fontsize=fs, usetex=False)

    [
        axe.axhline(
            0, color="red", lw=5, alpha=0.5
        )
        for axe in ax[:, 0].flatten()
    ]
    p = 0
    n = 9
    # loop through the bright, int, faint
    bins = [0, 3, 7, 10]
    for i in range(3):
        for code in codes:
            mean_j = []
            std_j = []
            out_j = []
            
            for band in bands:
                mean_j.append(np.mean(means[code][band][bins[i]:bins[i+1]]))

                std_j.append(np.mean(stds[code][band][bins[i]:bins[i+1]]))
                out_j.append(np.mean(outs[code][band][bins[i]:bins[i+1]]))
            ax[p, 0].plot(
                xs[:n],
                mean_j[:n],
                marker=symbols[code],
                color=colors[code],
                label=labels[code],
                markersize=10,
            )
            ax[p, 1].plot(
                xs[:n],
                std_j[:n],
                marker=symbols[code],
                color=colors[code],
                label=labels[code],
                markersize=10,
            )
            ax[p, 2].plot(
                xs[:n],
                out_j[:n],
                marker=symbols[code],
                color=colors[code],
                label=labels[code],
                markersize=10,
            )
            ax[0, 0].legend(fontsize=fs+2, ncol=legend_columns, loc=(0, 1.2))

        p += 1
    
    ax[0, 0].set_ylabel('Bright galaxies \n \n $\mathrm{{Pred_{{b/t}}}} - \mathrm{{True_{{b/t}}}} $', fontsize=23, usetex=False)
    ax[1, 0].set_ylabel('Intermediate galaxies \n \n $\mathrm{{Pred_{{b/t}}}} - \mathrm{{True_{{b/t}}}} $', fontsize=23, usetex=False)
    ax[2, 0].set_ylabel('Faint galaxies \n \n $\mathrm{{Pred_{{b/t}}}} - \mathrm{{True_{{b/t}}}} $', fontsize=23, usetex=False)
    fig.canvas.draw()
    [ax.set_yticklabels(ax.get_yticklabels(), fontsize=19, usetex=False) for ax in ax.flatten()]
    [axe.set_xticks(np.arange(len(bands))) for axe in ax.flatten()]
    band_labels = [labels[band] for band in bands]
    [axe.set_xticklabels(band_labels, rotation=45, fontsize=22, usetex=False) for axe in ax.flatten()]
    if len(bands)==9:
        [axe.add_artist(Rectangle([-0.2, -1], 3.5, 3, fill=True, alpha=0.05, color='red')) for axe in ax.flatten()]
        [axe.add_artist(Rectangle([3.7, -1], 0.55, 3, fill=True, alpha=0.05, color='blue')) for axe in ax.flatten()]
        [axe.add_artist(Rectangle([4.7, -1], 0.6, 3, fill=True, alpha=0.05, color='red')) for axe in ax.flatten()]
        [axe.add_artist(Rectangle([5.8, -1], 2.38, 3, fill=True, alpha=0.05, color='blue')) for axe in ax.flatten()]
    return fig


def summary_plot2D(codes, param, summary2D, mag_bins, bt_bins, labels):

    """
    Create the 2D Summary plot representing the Bias and  Dispersion for a parameter
    regarding the magnitude and the bulge over total flux ratio.

    ## TO DO: remove codes, which could be automatically found in the summary 2D summary.

    Parameters
    ----------

    codes : list of string
        The list of codes names to be plotted.

    param : string
        The parameter name to be plotted.

    summary2D : List of dictionaries, of length 2
        The Bias (summary2D[0]) and dispersion (summary2D[1]) for each code
        Each metric is a dictionary of 2D numpy arrays, with the metrics regarding mag and bt, for each code.

    mag_bins : numpy array
        The bins of magnitude in which the 2D summary has been computed on.

    bt_bins : numpy array
        The bins of bulge over total flux ratio in which the 2D summary has been computed on.

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    fs_colorbar = 30
    std_cmap = "autumn"
    fig, ax = plt.subplots(len(codes), 2, figsize=(20, 35))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    fig.suptitle(labels[param], y=0.92, fontsize=40)
    for i, code in enumerate(codes):
        divider = make_axes_locatable(ax[i, 0])
        cax1 = divider.append_axes("right", size="8%", pad=0.9)
        cax2 = divider.append_axes("right", size="8%", pad=0.11)
        nb_bins_bt = np.shape(summary2D[0][param][code])[1]
        nb_bins_mag = np.shape(summary2D[0][param][code])[2]
        x, y = np.meshgrid(np.arange(nb_bins_bt) + 0.5, np.arange(nb_bins_mag) + 0.5)
        if (code in ["gala", "profit"]) & (param == "re"):
            bulges = ax[i, 0].pcolormesh(
                summary2D[0][param][code][0],
                edgecolors="w",
                cmap="Blues",
                norm=LogNorm(),
            )
            std_bulges = ax[i, 0].scatter(
                x,
                y,
                s=100,
                c=summary2D[1][param][code][0],
                cmap=std_cmap,
                norm=LogNorm(),
            )
            min_cbar, max_cbar = np.log10(
                np.min(summary2D[0][param][code][0])
            ), np.log10(np.max(summary2D[0][param][code][0]))
            cbar_ticks = np.logspace(min_cbar, max_cbar, 8)
            min_cbar_std, max_cbar_std = np.log10(
                np.min(summary2D[1][param][code][0])
            ), np.log10(np.max(summary2D[1][param][code][0]))
            cbar_std_ticks = np.logspace(min_cbar_std, max_cbar_std, 8)
        else:
            bulges = ax[i, 0].pcolormesh(
                summary2D[0][param][code][0], edgecolors="w", cmap="Blues"
            )
            std_bulges = ax[i, 0].scatter(
                x, y, s=100, c=summary2D[1][param][code][0], cmap=std_cmap
            )
            min_cbar, max_cbar = np.min(summary2D[0][param][code][0]), np.max(
                summary2D[0][param][code][0]
            )
            min_cbar_std, max_cbar_std = np.min(summary2D[1][param][code][0]), np.max(
                summary2D[1][param][code][0]
            )
            cbar_ticks = np.linspace(min_cbar, max_cbar, 8)
            cbar_std_ticks = np.linspace(min_cbar_std, max_cbar_std, 8)

        ax[i, 0].set_yticks(np.linspace(0, nb_bins_mag, nb_bins_mag + 1))
        ax[i, 0].set_yticklabels(np.round(mag_bins, 1), fontsize=26, usetex=False)
        ax[i, 0].set_xticks(np.linspace(0, nb_bins_bt, nb_bins_bt + 1)[:-1])
        ax[i, 0].set_xticklabels(np.round(bt_bins[:-1], 1), fontsize=26, rotation=30, usetex=False)
        ax[i, 0].set_title(f"{labels[code]} Bulges", fontsize=30, usetex=False)
        ax[i, 0].set_xlabel(r"$I_{\mathrm{\mathsf{E}}}$ true b/t", fontsize=29, usetex=False)
        ax[i, 0].set_ylabel(r"$I_{\mathrm{\mathsf{E}}}$ true magnitude", fontsize=33, usetex=False)

        cb = plt.colorbar(bulges, ax=ax[i, 0], cax=cax1)
        cb.ax.set_title(r"$|\tilde{{\mathcal{B}}}|$", x=0.5, y=1.05, fontsize=fs_colorbar, usetex=False)
        cb.ax.yaxis.set_ticks(cbar_ticks)
        cb.ax.yaxis.set_ticklabels(cbar_ticks, fontsize=25)
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        cb_std = plt.colorbar(std_bulges, ax=ax[i, 0], cax=cax2)
        cb_std.ax.yaxis.set_ticks(cbar_std_ticks)
        cb_std.ax.yaxis.set_ticklabels(cbar_std_ticks, fontsize=25)
        cb_std.ax.set_title(r"$\tilde{{\mathcal{D}}}$", x=0.6, y=1.05, fontsize=fs_colorbar, usetex=False)
        cb_std.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        """ DISKS """
        divider = make_axes_locatable(ax[i, 1])
        cax1 = divider.append_axes("right", size="8%", pad=0.7)
        cax2 = divider.append_axes("right", size="8%", pad=0.1)

        disks = ax[i, 1].pcolormesh(
            summary2D[0][param][code][1], edgecolors="w", cmap="Blues"
        )  # , vmin=minv, vmax=maxv)
        std_disks = ax[i, 1].scatter(
            x, y, s=100, c=summary2D[1][param][code][1], cmap=std_cmap
        )

        ax[i, 1].set_yticks(np.linspace(0, nb_bins_mag, nb_bins_mag + 1))
        ax[i, 1].set_yticklabels(np.round(mag_bins, 1), fontsize=26, usetex=False)
        ax[i, 1].set_xticks(np.linspace(0, nb_bins_bt, nb_bins_bt + 1)[:-1])
        ax[i, 1].set_xticklabels(np.round(bt_bins[:-1], 1), fontsize=26, rotation=30, usetex=False)
        
        ax[i, 1].set_title(f"{labels[code]} Disks", fontsize=30, usetex=False)
        ax[i, 1].set_xlabel(r"$I_{\mathrm{\mathsf{E}}}$ true b/t", fontsize=29, usetex=False)
        ax[i, 1].set_ylabel(r"$I_{\mathrm{\mathsf{E}}}$ true magnitude", fontsize=33, usetex=False)

        cb2 = plt.colorbar(disks, ax=ax[i, 1], cax=cax1)
        cb2.ax.set_title(r"$|\tilde{{\mathcal{B}}}|$", x=0.5, y=1.05, fontsize=fs_colorbar, usetex=False)
        cb2_ticks = np.linspace(
                np.min(summary2D[0][param][code][1]),
                np.max(summary2D[0][param][code][1]),
                8,
            )
        cb2.ax.yaxis.set_ticks(cb2_ticks)
        cb2.ax.yaxis.set_ticklabels(cb2_ticks, fontsize=25)

        cb2.ax.yaxis.set_ticks_position("left")
        cb2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb2_std = plt.colorbar(std_disks, ax=ax[i, 1], cax=cax2)
        cb2_std_ticks = np.linspace(
                np.min(summary2D[1][param][code][1]),
                np.max(summary2D[1][param][code][1]),
                8,
            )
        cb2_std.ax.yaxis.set_ticks(cb2_std_ticks)
        cb2_std.ax.yaxis.set_ticklabels(cb2_std_ticks, fontsize=25)
        cb2_std.ax.set_title(r"$\tilde{{\mathcal{D}}}$", x=0.6, y=1.05, fontsize=fs_colorbar, usetex=False)
        cb2_std.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    return fig


def photo_trumpet_plots(cats, codes, fields, LABELS, TU_std, compo=None, nb_free=None):
    
    for field in fields:
        nb_lines = int(np.ceil(len(codes) / 2))
        fig, ax = plt.subplots(nb_lines, 2, figsize=(7*2, 7*(nb_lines-nb_lines/4)))
        ax = ax.flatten()
        fst=18
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        p = 0
        for code in codes:
            cat = cats[f'{code}_{field}']
            a = ax[p].scatter(cat[:, 0], cat[:, 1], c=cat[:, 2], marker='.', cmap='gist_rainbow', s=1)
            ax[p].set_ylim([-1, 1])
            plt.colorbar(a, ax=ax[p]).set_label(label='BT', size=20)
            if compo in ['bulge', 'disk']:
                if nb_free:
                    ax[p].set_title(f'{LABELS[code]}, \n Field {field}, {compo} component, bulge free', fontsize=fst)
                else:
                    ax[p].set_title(f'{LABELS[code]}, \n Field {field}, {compo} component, bulge fixed', fontsize=fst)
            elif compo == 'total':
                if nb_free:
                    ax[p].set_title(f'{LABELS[code]}, \n Field {field}, entire galaxy, bulge free', fontsize=fst)
                else:
                    ax[p].set_title(f'{LABELS[code]}, \n Field {field}, entire galaxy, bulge fixed', fontsize=fst)
            else:
                ax[p].set_title(f'{LABELS[code]}, \n Field {field}', fontsize=fst)
            
            xbins = np.linspace(14, 26, 11)
            xbins_plot = np.linspace(14, 26, 11)
            means = []
            stds = []
            for i, mag in enumerate(xbins[:-1]):
                if i == 0:
                    continue
                indices_mag = np.where((cat[:, 0] > mag)  & (cat[:, 0]  < xbins[i+1]))[0]
                sub_cat = cat[indices_mag]
                if len(sub_cat) == 0:
                    xbins_plot = np.delete(xbins_plot, i)
                    continue
                indices_in = np.where(sub_cat[:, 1] < 5 * TU_std['vis'][i])
                sub_cat = sub_cat[indices_in]
                mean = np.nanmean(sub_cat[:, 1])
                std = np.nanstd(sub_cat[:, 1])
                means.append(mean)
                stds.append(std)

            ax[p].errorbar(xbins_plot[2:-1], means[1:], stds[1:], c='black')
            ax[p].set_xlabel('$I_{\mathrm{\mathsf{E}}}$ true magnitude', fontsize=20)
            ax[p].set_ylabel("$\delta f(I_{\mathrm{\mathsf{E}}})$", fontsize=20)
            ax[p].tick_params(axis='both', labelsize=15)
            ax[p].set_xlim([16, 26])
            p += 1
    
        if len(codes) % 2 != 0:
            ax[-1].set_visible(False)
        st.pyplot(fig)
    return 0

def photo_trumpet_plots_multi_band(cats, codes, bands, LABELS, TU_std, compo, nb_free):
    
    for band in bands:
        nb_lines = int(np.ceil(len(codes) / 2))
        print(nb_lines)
        fig, ax = plt.subplots(nb_lines, 2, figsize=(7*(nb_lines+1), 7*nb_lines))
        ax = ax.flatten()
        plt.subplots_adjust(hspace=0.4)
        p = 0
        fst=18
        for code in codes:
            cat = cats[f'{code}_{band}']
            a = ax[p].scatter(cat[:, 0], cat[:, 1], c=cat[:, 2], marker='.', cmap='gist_rainbow', s=1)
            ax[p].set_ylim([-1, 1])
            plt.colorbar(a, ax=ax[p]).set_label(label='BT', size=20)
            if compo in ['bulge', 'disk']:
                if nb_free:
                    ax[p].set_title(f'{LABELS[code]}, \n $\mathrm{{{band}}}$, {compo} component, bulge free', fontsize=fst)
                else:
                    ax[p].set_title(f'{LABELS[code]}, \n $\mathrm{{{band}}}$, {compo} component, bulge fixed', fontsize=fst)
            elif compo == 'total':
                if nb_free:
                    ax[p].set_title(f'{LABELS[code]}, \n $\mathrm{{{band}}}$, entire galaxy, bulge free', fontsize=fst)
                else:
                    ax[p].set_title(f'{LABELS[code]}, \n  $\mathrm{{{band}}}$, entire galaxy, bulge fixed', fontsize=fst)

            xbins = np.linspace(14, 26, 11)
            xbins_plot = np.linspace(14, 26, 11)
            means = []
            stds = []
            for i, mag in enumerate(xbins[:-1]):
                if i == 0:
                    continue
                indices_mag = np.where((cat[:, 0] > mag)  & (cat[:, 0]  < xbins[i+1]))[0]
                sub_cat = cat[indices_mag]
                if len(sub_cat) == 0:
                    xbins_plot = np.delete(xbins_plot, i)
                    continue
                indices_in = np.where(sub_cat[:, 1] < 5 * TU_std['vis'][i])
                sub_cat = sub_cat[indices_in]
                mean = np.nanmean(sub_cat[:, 1])
                std = np.nanstd(sub_cat[:, 1])
                means.append(mean)
                stds.append(std)

            # print(len(xbins), len(means))
            ax[p].errorbar(xbins_plot[2:-1], means[1:], stds[1:], c='black')
            ax[p].set_xlabel('$I_{\mathrm{\mathsf{E}}}$ true magnitude', fontsize=20)
            ax[p].set_ylabel("$I_{\mathrm{\mathsf{E}}}$  $\delta f$", fontsize=20)
            ax[p].tick_params(axis='both', labelsize=15)
            ax[p].set_xlim([16, 27])
            p += 1
    
        if len(codes) % 2 != 0:
            ax[-1].set_visible(False)
        st.pyplot(fig)
    return 0