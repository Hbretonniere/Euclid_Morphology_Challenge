# EUCLID Morpho Challenge Analysis

Welcome the Euclid MorphoChallenge Do It Yourself Plotting platform ! \
The goal of this platform is to reproduce most of the Euclid Morpho Challenge paper plots, \
but with a great control of the different parameters and definitions.

---------------------------------

## Installation
In addition to standard libraries such as numpy or matplotlib, you will need streamlit >= v1.x:

```bash
pip install streamlit
```
---------------------------------
## Data
```bash
cd data
Download with zenodo (# TO DO)
```
---------------------------------
## Run the App
```bash
streamlit run streamlit_app.py
```
A web page will open in your browser, and you will be able to play with the data.

---------------------------------

## App README

Here is a description of how the app is organized, and what you can do. You can find it also directly on the webpage.
---------------------------------


### On the left panel, you can select the different things you want to plot:


- First, the dataset you want to inspect: Single or double Sérsic mono band simulation,
realistic morphologies (simulated with deep learning), or multiband data. If you chose
the latest, another button will appear to ask you which band you want to see the fitting.
- Then, the type of plot. It will differ following the chosen dataset. You can find the description of the figure
just above it, on the right panel.
- Then, you can chose the structural parameters you want to chose. You can also click on the 'Plot all'
check box to plot all the available parameters.
- Then, same but for the softwares you want to study. Here again, it will only show you the
softwares available for the type of simulation your looking at.
- Finally, you can chose the number of bin for the different plots.
- You can try the 'more options' button, but it is still in beta mode.


---------------------------------

### On the Right panel, the plot and some plotting option will appear:
- Above this README, that you hide, you have a 'Demo Version' checkbox. If it \
selected, it will do the study with only 1/100th of the catalogues. It is \
to see how the platform works. When you have selected the right setup, be sure to de select it !\
- Then, you have a slider that can change the Outlier Fraction defined in the paper: \
if you increase it, you will less (resp. more) outliers (weaker (resp. stronger) outlier condition). \
- Then, another slider(s) to control the magnitude range (x-axis) of the plots. If you are in \
2D summary plot mode, you will have a second one to control the b/t range.\
- In some plot modes, below the figure, you have a 'Save results' checkbox. If you check it, it will save\
results as a pickle, with all the information. To directly save the figure, you will have to right click and save.\
- Finally, in the  'Summary Score' plot mode, you will have 4 new sliders. \
They control the weights of the different metric on the global score. You can also chose \
remove the SNR weights of the bins in $\mathcal{S}$ by checking the new checkbox \
which is just above the figure."
