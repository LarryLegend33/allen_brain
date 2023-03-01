import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as pl
import copy
from random import shuffle
import itertools
import math

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

    
def generate_snmc_connectivity(assembly_size, num_assemblies, num_particles, snmc_type, *brain_regions):

    """ Synapses Per Neuron """

    # In original, WTA inhibitory neuron hits all other WTA inhib and excitatory cells. Inhibitory WTA neurons
    # project down to Layer V MUX for both P and Q. 
    
    if snmc_type == "original":
        # WTA
        excitatory_wta_to_wta = 1 
        excitatory_wta_intracortical = 1 # state travels to next microcolumn.
        inhibitory_wta_to_wta = 2 * (num_assemblies - 1) # hits all other wta excitatory neurons
        excitatory_wta_to_scoring = 0  # one to mux.
        inhibitory_wta_to_scoring = 2
        # Assemblies
        assembly_to_wta = 1
        assembly_to_scoring = 2
        # Scoring
        q_accumulator_to_scoring = 1
        q_accumulator_to_norm = 1
        q_thresholder_to_scoring = 2
        q_mux_to_scoring = 1
        p_accumulator_to_scoring = 1
        p_thresholder_to_scoring = 2
        p_mux_to_norm = 1
        inhibitory_mux_to_mux = 0  # one for each mux

    # In updated SNMC bioreal, the WTA excitatory AND inhibitory neuron receive input from assemblies at the same time.
    # The inhibitory neuron projects to all other WTA excitatory neurons. The WTA excitatory neuron projects down to a
    # neuron in Layer V that inhibits the dendrites of the P and Q mux for losing values. This architecture prevents
    # the two synapse delay in inhibiting losing values in the WTA circuit, and prevents the need for inhibition of
    # losing WTA inhibitory neurons. Scoring should be cleaner because only one WTA should ever activate. 
        
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
        q_thresholder_to_scoring = 2
        q_mux_to_scoring = 1
        p_accumulator_to_scoring = 1
        p_thresholder_to_scoring = 2
        p_mux_to_norm = 1
        inhibitory_mux_to_mux = 2  # one for each mux

    """ Build a dictionary that establishes a mapping between components """ 
        
    connections = {}
    connections["Scoring", "Scoring"] = q_accumulator_to_scoring + q_accumulator_to_norm + q_thresholder_to_scoring + q_mux_to_scoring + p_accumulator_to_scoring + p_thresholder_to_scoring + p_mux_to_norm + (num_assemblies * inhibitory_mux_to_mux)
    connections["WTA", "WTA"] = num_assemblies * (excitatory_wta_to_wta + inhibitory_wta_to_wta)
    connections["WTA", "Scoring"] = num_assemblies * (inhibitory_wta_to_scoring + excitatory_wta_to_scoring)
    connections["WTA", "DownstreamVariable"] = num_assemblies * excitatory_wta_intracortical
    connections["Assemblies", "WTA"] = num_assemblies * assembly_size * assembly_to_wta
    connections["Assemblies", "Scoring"] = num_assemblies * assembly_size * assembly_to_scoring
    connections.update((k, v*num_particles) for k, v in connections.items())

    connections["Scoring", "Normalizer"] = 2 * num_particles
    connections["Normalizer", "Resampler"] = num_particles
    connections["Resampler", "ObsStateSync"] = num_particles
    connections["ObsStateSync", "WTA"] = num_particles

    """ Randomly assign components to relevant brain regions """
    
    if brain_regions == ():
         brain_regions = ["V1L2", "V1L4", "V1L5", "CP", "GPi", "LD"]
         shuffle(brain_regions)
    else:
        brain_regions = brain_regions[0]
        
    snmc_components = ["Assemblies", "WTA", "Scoring", "Normalizer",
                       "Resampler", "ObsStateSync"]
    random_brain_snmc_pairings = dict(zip(brain_regions, snmc_components))
    random_brain_snmc_pairings["S1"] = "DownstreamVariable"
    pairwise_synapses = [(a[0], a[1]) for a in list(itertools.product(brain_regions + ["S1"], brain_regions + ["S1"]))]

    """ Project the component mapping onto the random brain regions """ 
    
    def map_brain_to_snmc(source, termination):
        try:
            source_brain = random_brain_snmc_pairings[source]
            termination_brain = random_brain_snmc_pairings[termination]
            return connections[source_brain, termination_brain]
        except KeyError:
            return 0

    synapse_dictionary = {}
    for pw_syn in pairwise_synapses:
        synapse_dictionary[pw_syn[0], pw_syn[1]] = map_brain_to_snmc(pw_syn[0], pw_syn[1])

    """ Compress the resolution of the random mapping to the figure """

    compressed_synapse_dictionary = {}
    cp_to_v1_accumulator = 0
    gpi_to_v1_accumulator = 0
    thal_to_v1_accumulator = 0
    v1l5_to_v1_accumulator = 0
    v1l4_to_v1_accumulator = 0
    v1l2_to_v1_accumulator = 0
    
    for k, v in synapse_dictionary.items():
        if k[0] == "CP" and k[1][0:2] == "V1":
            cp_to_v1_accumulator += v
        elif k[0] == "GPi" and k[1][0:2] == "V1":
            gpi_to_v1_accumulator += v
        elif k[0] == "LD" and k[1][0:2] == "V1":
            thal_to_v1_accumulator += v
        elif k[0] == "V1L5" and k[1][0:2] == "V1":
            v1l5_to_v1_accumulator += v
        elif k[0] == "V1L4" and k[1][0:2] == "V1":
            v1l4_to_v1_accumulator += v
        elif k[0] == "V1L2" and k[1][0:2] == "V1":
            v1l2_to_v1_accumulator += v
        else:
            compressed_synapse_dictionary[k] = v

    compressed_synapse_dictionary["CP", "V1"] = cp_to_v1_accumulator
    compressed_synapse_dictionary["GPi", "V1"] = gpi_to_v1_accumulator
    compressed_synapse_dictionary["LD", "V1"] = thal_to_v1_accumulator
    compressed_synapse_dictionary["V1L5", "V1"] = v1l5_to_v1_accumulator
    compressed_synapse_dictionary["V1L4", "V1"] = v1l4_to_v1_accumulator
    compressed_synapse_dictionary["V1L2", "V1"] = v1l2_to_v1_accumulator

    return synapse_dictionary, compressed_synapse_dictionary

def create_realsynapse_dictionary(connectivity_df):
    realsyn_dict = {}
    termini = ["V1", "CP", "GPi", "LD"]
    sources = connectivity_df["Source"]
    for source in sources:
        source_row = connectivity_df.loc[connectivity_df["Source"] == source]
        for terminus in termini:
            realsyn_dict[source, terminus] = source_row[terminus].values[0]
    realsyn_dict_filtered = {}
    for k, v in realsyn_dict.items():
        if not math.isnan(v):
            realsyn_dict_filtered[k] = v
    return realsyn_dict_filtered


def smc_barplot_from_dicts(real_df, assembly_size, num_assemblies,
                           num_particles, syns_to_exclude):
    # normalize all values (i.e. total synapses)
    realsyn_dict = create_realsynapse_dictionary(real_df)
    for syn in syns_to_exclude:
        del realsyn_dict[syn]
    reference_snmc_orig = generate_snmc_connectivity(
        assembly_size, num_assemblies, num_particles,
        "original", ["V1L4", "V1L2", "V1L5", "CP", "GPi", "LD"])[1]
    reference_snmc_update = generate_snmc_connectivity(
        assembly_size, num_assemblies, num_particles,
        "update", ["V1L4", "V1L2", "V1L5", "CP", "GPi", "LD"])[1]
    random_snmc = generate_snmc_connectivity(assembly_size, num_assemblies, num_particles,
                                             "update")[1]

    realkeys = realsyn_dict.keys()
    network_labels = ["Real", "SNMC", "SNMC_Update", "Shuffled"]
    norm_syn_dicts = []
    all_syn_dicts = [realsyn_dict, reference_snmc_orig, reference_snmc_update, random_snmc]
    for d in all_syn_dicts:
        syn_dict = {k: d[k] for k in realkeys}
        total_synapses = np.sum(list(syn_dict.values()))
        syn_dict.update((k, v / total_synapses) for k, v in syn_dict.items())
        norm_syn_dicts.append(syn_dict)

    x = [k[0]+"-"+k[1] for k in realkeys]
    y = [norm_syn_dicts[0][k] for k in realkeys]
    hue = np.repeat(network_labels[0], len(realkeys)).tolist()
    
    for k, v in realsyn_dict.items():
        print(k)
        x += np.repeat(k[0]+"-"+k[1], 3).tolist()
        y += [norm_syn_dicts[1][k], norm_syn_dicts[2][k], norm_syn_dicts[3][k]]
        hue += network_labels[1:]
    cpal = sns.color_palette("Set2")

    fig, ax = pl.subplots(1, 1)
    sns.barplot(x=x, y=y, hue=hue, palette=cpal, ax=ax)
    ax.set_xlabel("Synapse")
    ax.set_ylabel("Proportion of Circuit Connectivity")
    ax.tick_params(axis='x', labelsize=8)
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

#syns_to_exclude = ["CP-CP", "GPi-GPi", "LD-LD"]
#scatter_and_barplot_smc(longrange_bar, "Synapse", "NormedToSum", "NetworkType", syns_to_exclude, "Synapse")
   
   
