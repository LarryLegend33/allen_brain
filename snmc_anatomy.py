import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as pl
import copy
from random import shuffle

# note that longrange_bar is currently normed to V1 self connectivity because
# that is a valid reference in a Syn-EGFP experiment. However, the problem of norming everything to that
# is that SNMC_Update and SNMC have different values for inter-V1. The relative amounts to V1 connections are correct.
# I think it should be normed by the ratio between the sum of SNMC and the sum of real. 

longrange = pd.read_csv("longrange.csv")
longrange_bar = pd.read_csv("longrange_df.csv")
microcircuit = pd.read_csv("microcircuit.csv")
microcircuit_probability = pd.read_csv("microcircuit_probabilities.csv")
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
    
def generate_snmc_connectivity(assembly_size, num_assemblies, num_particles, snmc_type,
                               *anatomical_dict):
    if snmc_type == "update":
        # WTA
        excitatory_wta_to_wta = 0 
        excitatory_wta_intracortical = 1 # state travels to next microcolumn.
        inhibitory_wta_to_wta = num_assemblies - 1 # hits all other wta excitatory neurons
        excitatory_wta_to_scoring = 1  # one to mux.
        inhibitory_wta_to_scoring = 0
        # Assemblies
        assembly_to_wta = 2
        assembly_to_scoring = 2
        # Scoring
        q_accumulator_to_scoring = 1
        q_accumulator_to_norm = 1
        p_mux_to_norm = 1
        q_thresholder_to_scoring = 2
        p_thresholder_to_scoring = 2
        inhibitory_mux_to_mux = 2  # one for each mux
        
    # what is the probability of neurons being connected? have to factor in prob of drawing both neurons 

    scoring_internal = q_accumulator_to_scoring + q_accumulator_to_norm + p_mux_to_norm + q_thresholder_to_scoring + p_thresholder_to_scoring + (num_assemblies * inhibitory_mux_to_mux)
    wta_internal = excitatory_wta_to_wta + inhibitory_wta_to_wta
    wta_to_scoring = inhibitory_wta_to_scoring + excitatory_wta_to_scoring
    
    total_microcircuit_synapses = map(lambda x: x * num_particles, [scoring_internal, wta_internal, wta_to_scoring, assembly_to_wta, assembly_to_scoring])
    
    scoring_to_normalizer = 2 * num_particles
    normalizer_to_resampler = num_particles
    resampler_to_obs_sync = num_particles
    obs_sync_to_sampler = num_particles

    synapses = ["V1", "CP", "GPi", "Thal"]
    pairwise_synapses = [a[0] + "-" + a[1] for a in list(itertools.permutations(synapses, 2))] + ["V1-V1", "V1-S1"]

    brain_regions = ["V1L2", "V1L4", "V1L5", "S1", "CP", "GPi", "Thal"]
    shuffle(brain_regions)
    snmc_components = ["Assemblies", "WTA", "Scoring", "Normalizer", "Resampler", "ObsStateSync", "DownstreamVariable"]

    random_brain_snmc_pairings = zip(brain_regions, snmc_components)
    
    



    
    
    if anatomical_dict == ():
        anatomical_dict = { "


        
    else:
        anatomical_dict = anatomical_dict[0]

    total_connections = total_microcircuit_synapses + scoring_to_normalizer + normalizer_to_resampler + resampler_to_obs_sync + obs_sync_to_sampler
    
    
        
    return anatomical_dict
    
    
                         
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

#syns_to_exclude = ["CP-CP", "GPi-GPi", "LD-LD"]
#scatter_and_barplot_smc(longrange_bar, "Synapse", "NormedToSum", "NetworkType", syns_to_exclude, "Synapse")
   
    
