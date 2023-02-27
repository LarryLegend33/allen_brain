import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as pl
import copy

# note that longrange_bar is currently normed to V1 self connectivity because
# that is a valid reference in a Syn-EGFP experiment. However, the problem of norming everything to that
# is that SNMC_Update and SNMC have different values for inter-V1. The relative amounts to V1 connections are correct.
# I think it should be normed by the ratio between the sum of SNMC and the sum of real. 

longrange = pd.read_csv("longrange.csv")
longrange_bar = pd.read_csv("longrange_df.csv")
microcircuit = pd.read_csv("microcircuit.csv")
microcircuit_probability = pd.read_csv("microcircuit_probabilities.csv")s
microcircuit_probability_bar = pd.read_csv("microcircuit_probs_df.csv")

def plot_smc_heatmap(df, col_to_exclude):
    network_types = np.unique(df["NetworkType"]).tolist()
    fig, axes = pl.subplots(1, len(network_types))
    for (network_type, ax) in zip(network_types, axes):
        df_network = df[df["NetworkType"] == network_type]
        df_filt = df_network.loc[:, ~df_network.columns.isin([col_to_exclude, "NetworkType"])]
        sns.heatmap(df_filt, yticklabels=df_network[col_to_exclude], ax=ax, cmap="viridis")
        ax.set_ylabel("Source")
        ax.set_xlabel("Termination Zone")
        ax.set_title(network_type)
    pl.show()
                         
def scatter_and_barplot_smc(df, xval, yval, hueval, syns_to_exclude, exclude_flag):
    cpal = sns.color_palette("Set2")
    df_filt = df[~df[exclude_flag].isin(syns_to_exclude)]
    fig, ax = pl.subplots(2, 1)
    sns.barplot(data=df_filt, x=xval, y=yval, hue=hueval, palette=cpal, ax=ax[0])
    synapses = df_filt["Synapse"].unique()
    scatter_realvals = df_filt[df_filt["NetworkType"] == "Real"]
    scatter_smc = df_filt[df_filt["NetworkType"] == "SNMC_Update"]
    x = [scatter_realvals[scatter_realvals["Synapse"] == syn]["NormedToSum"].values[0] for syn in synapses]
    y = [scatter_smc[scatter_smc["Synapse"] == syn]["NormedToSum"].values[0] for syn in synapses]
    sns.scatterplot(x=x, y=y, ax=ax[1], color=cpal[3])
    ax[1].set_xlabel("Real Connection Strength")
    ax[1].set_ylabel("SNMC Connection Strength")
    y_offset = 0
    for xloc, yloc, syn in zip(x, y, synapses):
        if xloc == 0 and yloc == 0:
            y_off = copy.deepcopy(y_offset)
            y_offset += .05
            print(syn)
        else:
            y_off = 0
        ax[1].annotate(syn, (xloc + .005, yloc + y_off))
        ax[1].set_xlim([-.05, np.max(x) + .1])
        ax[1].set_ylim([-.05, np.max(y) + .1])
    pl.show()


# want real normedconnection on x axis, snmc normedconnection on y axis PER connection. 

syns_to_exclude = ["CP-CP", "GPi-GPi", "LD-LD"]
scatter_and_barplot_smc(longrange_bar, "Synapse", "NormedToSum", "NetworkType", syns_to_exclude, "Synapse")
   
    
