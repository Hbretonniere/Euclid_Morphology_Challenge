import streamlit as st

readme = """
### Welcome the Euclid MorphoChallenge Do It Yourself Plotting plateform !

The goal of this platform is to enable the scientists to navigate through the results of the Euclid Morphology Challenge
with a control on the different parameters to maximise the feedback.

On the leftmost panel, you can select the different things you want to plot:
- the **dataset** you want to inspect: Single or double Sérsic mono-band simulation, realistic morphologies (simulated with deep learning), or multiband data.  
   If you chose double Sérsic, another button will appear to ask you which band you want to see the fitting.
- the **type of plot**: it will differ following the chosen dataset (summary plots, trumpet plots, uncertainty...).
   You can find the description of the figure just above it, on the right panel.
- the **structural parameters**.   
   You can also click on the 'Plot all' check box to plot all the available parameters.
- the **software** that made the analysis.
   Here again, it will only show you the software available for the type of simulation your looking at.
- the x-axis, i.e. the parmeter against which you want to compute the metrics.
- the **number of bins** for the different plots.
- _The `more options` button is still in beta mode_.

---

On the main panel, the plots and some plotting options will appear.
- a `Demo Version` checkbox, below the title.  
    If selected, it will do the study with only 1% of the catalogue to see how the platform works.  
    Once you have selected the right setup, ***be sure to uncheck the box !***
- a slider that can change the "Outlier Fraction" defined in the paper.
- some sliders to control the **x-axis range** of the plots.  
    If you are in 2D summary plot mode, you will have a second one to control the b/t range.
- If you have selected only one parameter, you will have 3 additional sliders to control the y-axis ranges of the three columns
- a 'Save results' button will be available below the figure of the summary plots.  
    If pressed, it will save the results as a pickle file, with all the information.  
    To save the figure, use a right click and "Save as ...".

Finally, in the  'Summary Score' plot mode, you will have 4 new sliders.
They control the weights of the different metric on the global score. You can also chose
remove the SNR weights of the bins in $\mathcal{S}$ by checking the new checkbox
which is just above the figure.")
"""


def add_help_box():
    help_box = st.sidebar.expander("Additional info")

    help_box.markdown("**Magnitude range**")
    help_box.markdown(
        "The magnitude range selects all galaxies in that range to compute the figures"
    )

    help_box.markdown("**Outliers Fraction**")
    help_box.markdown("The outliers fraction requested here corresponds to...")


def add_help_box_outliers():
    help_box = st.sidebar.expander("Additional info of outliers")

    help_box.markdown("** Axis Ratio and Bulge to Disk Ratio**")
    help_box.markdown(
        "For q and bt, a galaxy is defined as an outlier if its normalized  \
        error $$|\dfrac{\mathrm{True}  - \mathrm{Pred}}{\mathrm{range}}|$$ is greater than the outlier fraction defined above. \
        We use this definition considering that the outlier fraction must not depend on the value of the \
        parameter (making an error of $0.1$ on the fit of q for a galaxy with a True $q=0.2$ or $q=0.8$ should be considered \
        equally problematic. \n Because the range of q and bt are both 1, the normalized error is just the error.\
        Thus, for example with a $0.5$ outlier fraction, a galaxy with $q=0.3$ is considered as an outlier if its \
        if it is fitted with an error greater than $0.5$"
    )

    help_box.markdown("** MAGNITUDE AND RADII **")
    help_box.markdown(
        "For magnitude and radii, a galaxy is defined as an outlier if its relative  \
        error $|\dfrac{\mathrm{True} - \mathrm{Pred}}{\mathrm{True}}|$ is greater than the outlier fraction defined above. \
        We use this definition considering that the outlier fraction must depend on the value of the \
        parameter (making an error of $0.1$ on the fit of a galaxy with $r_\mathrm{e}=1$ is more problematic than \
        making the same error for a galaxy with $r_\mathrm{e}=10$ \
        For example, with a $0.5$ outlier fraction, a galaxy with radius 10 pixels is considered as an outlier \
        if its radius is fitted with an error greater than $5$ pixels. \
        For this same outlier fraction, a galaxy with radius $50$ pixels is is considered as an outlier \
        if its radius is fitted with an error greater than $25$ pixels."
    )

    help_box.markdown("** Sersic Index **")
    help_box.markdown(
        "For the sersic index, a galaxy is defined as an outlier if its normalized logarithmic  \
        error $|\dfrac{\mathrm{log(True)} - \mathrm{log(Pred)}}{\mathrm{log(range)}}|$ is greater than the outlier fraction defined above. \
        We use this definition considering that the outlier fraction must depend on the value of n \
        but this dependance is not linear. But doing an error of 1 for a galaxy with $n=6$ should be \
        considered less problematic than the same error for a galaxy with $n=1$\
        For example, with a $0.5$ outlier fraction, a galaxy with n=6 is considered as an outlier \
        if it is fitted with an error greater $2.5$. \
        For this same outlier fraction, a galaxy with $n=2$ is is considered as an outlier \
        if its radius is fitted with an error greater ~$1.2$."
    )
