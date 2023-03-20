import numpy as np
import seaborn as sns
import pandas as pd
import functools
from matplotlib import pyplot as pl
import copy
from random import shuffle
import itertools
import math
import fractions

# note that longrange_bar is currently normed to V1 self connectivity because
# that is a valid reference in a Syn-EGFP experiment. However, the problem of norming everything to that
# is that SNMC_Update and SNMC have different values for inter-V1. The relative amounts to V1 connections are correct.
# I think it should be normed by the ratio between the sum of SNMC and the sum of real. 

longrange = pd.read_csv("longrange.csv")
longrange_bar = pd.read_csv("longrange_df.csv")
microcircuit = pd.read_csv("microcircuit.csv")
microcircuit_probability = pd.read_csv("microcircuit_probabilities.csv")
microcircuit_probability_bar = pd.read_csv("microcircuit_probs_df.csv")

def make_v1_psp_reference():
    v1mouse = pd.read_csv("v1mouse.csv")
    v1_hm = v1mouse.loc[:, v1mouse.columns!="PreSyn"]
    fracmap = lambda x: fractions.Fraction(x, _normalize=False) if not x == "0/0" else np.nan
    v1_fractions = v1_hm.applymap(fracmap)
    v1_decimals = v1_fractions.applymap(np.float)
    v1_fractions["PreSyn"] = v1mouse["PreSyn"]
    compressed_dict = {}
    pyr_nodes = ["L2e", "L4e", "L5e"]
    inhib_nodes = ["L2i", "L4i", "L5i"]
    inh_suffixes = ["Pv", "Sst", "Vip"]
    ex_suffixes = ["", "ET", "IT"]
    mc_nodes = pyr_nodes + inhib_nodes
    compressed_df = pd.DataFrame(columns=mc_nodes)
    compressed_df["PreSyn"] = mc_nodes

#    problem here is you're adding too many connections
    
    def node_to_neuron(source_node, term_node):
        source_layer = source_node[0:2]
        source_identity = source_node[2:5]
        term_layer = term_node[0:2]
        term_identity = term_node[2:5]
        if source_identity == "e":
            source_rows = [source_layer+"Pyr"+es for es in ex_suffixes]
        elif source_identity == "i":
            source_rows = [source_layer+inh for inh in inh_suffixes]
        if term_identity == "e":
            term_columns = [term_layer+"Pyr"+es for es in ex_suffixes]
        elif term_identity == "i":
            term_columns = [term_layer+inh for inh in inh_suffixes]
        source_filtered = v1_fractions.loc[v1_fractions["PreSyn"].isin(source_rows)]
        term_filtered = source_filtered.loc[:, source_filtered.columns.isin(term_columns)]

        all_connections = functools.reduce(
            lambda a, b: a + b, [term_filtered[c].tolist() for c in term_filtered.columns])
        all_connections = list(filter(lambda a: type(a) == fractions.Fraction, all_connections))
        print(all_connections)        
        if len(all_connections) != 0:
            frac_connection = functools.reduce(
                lambda a, b: fractions.Fraction(
                    a.numerator + b.numerator, a.denominator + b.denominator, _normalize=False),
                all_connections)
            print(frac_connection)
        else:
            frac_connection = np.nan
        return frac_connection
    
    for (s, t) in itertools.product(mc_nodes, mc_nodes):
        compressed_dict[s, t] = node_to_neuron(s, t)
        compressed_df.loc[compressed_df["PreSyn"] == s, t] = node_to_neuron(s, t)

    # From Lefort et al. b/c Campagnola didn't record these synapses
    compressed_df.loc[compressed_df["PreSyn"] == "L4e", "L5e"] = .116
    compressed_df.loc[compressed_df["PreSyn"] == "L5e", "L4e"] = 0.0
    excitatory_df = compress_to_excitatory_only(compressed_df)
    agnostic_df = compress_to_layers_only(compressed_df)
    return v1_decimals, compressed_df, excitatory_df, agnostic_df

# here want to redo this in component space. make a random dictionary that goes through each iter.product of layers and indexes the
# assignment dictionary. 

def component_internals(num_assemblies, assembly_size, snmc_mapping):
    
    layers = ["L2", "L4", "L5"]
    components = ["WTA", "Assemblies", "Scoring"]
    component_df = pd.DataFrame(columns=components+["PreSyn"])
    component_df["PreSyn"] = components
    total_WTA = num_assemblies * 3
    assembly_autonorm_size = 3
    # one for p one for q
    total_Assemblies = 2 * (assembly_autonorm_size + (num_assemblies * assembly_size))
    mux_size = (2 * num_assemblies * assembly_size) + num_assemblies
    # have to resolve the dot that connects the outputs of the mux. that has to be a neuron. 
    ti_k_size = 2
    gate_size = 2
    # one for P one for Q
    total_Scoring = 2 * (mux_size + ti_k_size + gate_size)
    total_circuit_size = total_WTA + total_Assemblies + total_Scoring
    component_df.loc[component_df["PreSyn"] == "WTA", "WTA"] = ((num_assemblies / total_WTA) * (num_assemblies / total_WTA)) + 2 * ((num_assemblies / total_WTA) * (1 / total_WTA))
    # state feedback to assemblies.
    component_df.loc[component_df["PreSyn"] == "WTA", "Assemblies"] = (num_assemblies / total_WTA) * (
        (2 * (num_assemblies * assembly_size)) / total_Assemblies)
    component_df.loc[component_df["PreSyn"] == "WTA", "Scoring"] = (num_assemblies / total_WTA) * (assembly_size / total_Scoring) * 2
    component_df.loc[component_df["PreSyn"] == "Assemblies", "WTA"] = ((2 * (assembly_size * num_assemblies)) / total_Assemblies) * (1 / num_assemblies) * (num_assemblies / total_WTA) # normed to total theta gate neurons and only hits a single one. (= 1/ total_WTA)
    # all assembly neurons project to two autonorm inhibitory cells.
    # one inhibitory cell projects directly back to assemblies, the other to a tonic excitatory
    # neuron that excites the assemblies (almost identical to george's drawing "Autonorm_Multiply"
    component_df.loc[component_df["PreSyn"] == "Assemblies", "Assemblies"] = (2 * num_assemblies * assembly_size / total_Assemblies) * (4 / total_Assemblies) + (2 * (2 / total_Assemblies) * (2 * num_assemblies * assembly_size / total_Assemblies)) + (2 / total_Assemblies) * (1 / total_Assemblies)
    component_df.loc[component_df["PreSyn"] == "Assemblies", "Scoring"] = 2 * (((2 * assembly_size * num_assemblies) / total_Assemblies) * (1 / total_Scoring) + ((2 * assembly_size * num_assemblies) / total_Assemblies) * (1 / total_Scoring)) # assemblies to gate , assemblies to Mux twice.
    component_df.loc[component_df["PreSyn"] == "Scoring", "WTA"] = (1 / total_Scoring) * (num_assemblies / total_WTA)
    # want to clear the autonorm for the next step. ti_k to inhibitory assembly neurons (excitatory returns to tonic)
    component_df.loc[component_df["PreSyn"] == "Scoring", "Assemblies"] = (2 / total_Scoring) * (assembly_autonorm_size - 1) / total_Assemblies
    component_df.loc[component_df["PreSyn"] == "Scoring", "Scoring"] = (num_assemblies / total_Scoring) * (1 / total_Scoring) + (2 / total_Scoring) * (1 / total_Scoring) + (2 / total_Scoring) * (1/ total_Scoring) + (2 / total_Scoring) * (1/ total_Scoring) + ((2 * num_assemblies * assembly_size) / total_Scoring) * (1 / total_Scoring) + (2 * (num_assemblies * assembly_size / total_Scoring) * (1 / total_Scoring))
    # last is Mux to Mux where black nodes are a neuron.
    layer_df = pd.DataFrame(columns=layers + ["PreSyn"])
    layer_df["PreSyn"] = layers
    for pre, post in itertools.product(layers, layers):
        layer_df.loc[layer_df["PreSyn"] == pre, post] = component_df.loc[component_df["PreSyn"] == snmc_mapping["V1"+pre], snmc_mapping["V1"+post]].values[0]
    return component_df, layer_df


def compare_connectivity_scatter(real_df, snmc_df):
    fig, ax = pl.subplots(1, 1)
    xvals = []
    yvals = []
    for pre, post in itertools.product(["L2", "L4", "L5"], ["L2", "L4", "L5"]):
        xval = real_df.loc[real_df["PreSyn"] == pre, post].values[0]
        yval = snmc_df.loc[snmc_df["PreSyn"] == pre, post].values[0]
        xvals.append(float(xval))
        yvals.append(float(yval))
        ax.annotate(pre+"-" + post, (xval+.001, yval))
    cpal = sns.color_palette("Set2")
    print(xvals)
    print(yvals)
#    sns.scatterplot(xs=xvals, ys=yvals, ax=ax)# color=cpal[3])
    ax.scatter(x=xvals, y=yvals)
    ax.plot(range(5), range(5))
    ax.set_xlabel("Real")
    ax.set_ylabel("SNMC")
    ax.set_xlim([0, np.max(xvals)+.05])
    ax.set_ylim([0, np.max(yvals)+.05])
    pl.tight_layout()
    pl.show()


def compare_connectivity_heatmap(df_list, titles):
    maxval = np.max([np.nanmax(df.loc[:, ~df.columns.isin(["PreSyn"])].values) for df in df_list])
    fig, ax = pl.subplots(1, len(df_list))
    for i, (df, title) in enumerate(zip(df_list, titles)):
        if len(titles) > 1:
            sns.heatmap((df.loc[:, ~df.columns.isin(["PreSyn"])]).applymap(float),
                        yticklabels=df["PreSyn"], cmap="viridis", ax=ax[i], vmax=maxval)
            ax[i].set_title(title)
        else:
            sns.heatmap((df.loc[:, ~df.columns.isin(["PreSyn"])]).applymap(float),
                        yticklabels=df["PreSyn"], cmap="viridis", ax=ax, vmax=maxval)
            ax.set_title(title)
    pl.show()


def compress_to_excitatory_only(df):
    fig, ax = pl.subplots(1, 1)
    excit_pre = df[df["PreSyn"].isin(["L2e", "L4e", "L5e"])]
    excit = excit_pre.loc[:, excit_pre.columns.isin(["PreSyn", "L2e", "L4e", "L5e"])]
    return excit
    

def compress_to_layers_only(df):
# need the fractions here, or as a first pass just weight them evenly. (i.e. add the probs, divide by 4).
    layers = ["L2", "L4", "L5"]
# probability of l2 to l4, for example is prob 2e to 4e, 2e to 4i, 2i to 4e, 2i to 4i div by 4.
    layers_only_df = pd.DataFrame(columns=layers+["PreSyn"])
    layers_only_df["PreSyn"] = layers
    all_layer_combinations = itertools.product(layers, layers)
    excitatory_weight = .9
    inhibitory_weight = 1 - excitatory_weight
    weight_cartesian = list(itertools.product([excitatory_weight, inhibitory_weight], [excitatory_weight, inhibitory_weight]))
    unnormed_weights = list(map(lambda x: x[0] * x[1], weight_cartesian))
    weights = np.array(unnormed_weights) / np.sum(unnormed_weights)
    print(weights)
    for syn in all_layer_combinations:
        prob_e_e = df.loc[df["PreSyn"] == syn[0]+"e"][syn[1]+"e"].values[0]
        prob_e_i = df.loc[df["PreSyn"] == syn[0]+"e"][syn[1]+"i"].values[0]
        prob_i_e = df.loc[df["PreSyn"] == syn[0]+"i"][syn[1]+"e"].values[0]
        prob_i_i = df.loc[df["PreSyn"] == syn[0]+"i"][syn[1]+"i"].values[0]
        # fix by adding num and denom as accumulators 
        layers_only_df.loc[layers_only_df["PreSyn"] == syn[0], syn[1]] = np.sum(
            [a*b for a, b in zip(weights, [prob_e_e, prob_e_i, prob_i_e, prob_i_i])])
        
    # also, doing the second version of layers_only_df is hard because we add in the lefort data but its not accumulated -- its a float value. can go back into the dataset and get an n if its there. 
    return layers_only_df


def generate_snmc_macro_connectivity(assembly_size, num_assemblies, num_particles, randomize_brain_regions):
    # total axons that connect to the layer per circuit. agnostic to multiple connections within layer. 
    autonorm_size = 3
    # each theta gate projects to a wta. each wta to inhib wta.
    wta_to_wta = 3 * num_assemblies
    wta_to_assembly = num_assemblies * assembly_size
    # each wta goes to the mux once for p and q.
    wta_to_scoring = 2 * num_assemblies
    # all assemblies to to WTA
    assembly_to_wta = num_assemblies * assembly_size
    # all assemblies to to the autonorm. 2 autonorms project to all assemblies.
    # autonorms project internally * 2.
    assembly_to_assembly = num_assemblies * assembly_size + autonorm_size
    # all assemblies to scoring circuit in both P and Q. 
    assembly_to_scoring = 2 * num_assemblies * assembly_size
    # just the ready signal and the clearance of autonorm for scoring to other layers.  
    scoring_to_wta = 1
    scoring_to_assembly = 1
    # num_assemblies mux to TIK, two TIK to gate, two gate to gate, two TIK to TIK, assembly_size * num_assemblies + assembly_size * num_assemblies mux to mux. 
    scoring_to_scoring = num_assemblies + 2 + 2 + 2 + (2 * (assembly_size * num_assemblies))
    connections = {}
    connections["WTA", "WTA"] = wta_to_wta
    connections["WTA", "Assemblies"] = wta_to_wta
    connections["WTA", "Scoring"] = wta_to_scoring
    connections["WTA", "DownstreamVariable"] = num_assemblies
    connections["Assemblies", "WTA"] = assembly_to_wta
    connections["Assemblies", "Assemblies"] = assembly_to_assembly
    connections["Assemblies", "Scoring"] = assembly_to_scoring
    connections["Scoring", "WTA"] = scoring_to_wta
    connections["Scoring", "Assemblies"] = scoring_to_assembly
    connections["Scoring", "Scoring"] = scoring_to_scoring
    connections.update((k, v*num_particles) for k, v in connections.items())
    connections["Scoring", "Normalizer"] = 2 * num_particles
    connections["Normalizer", "Resampler"] = num_particles
    connections["Resampler", "ObsStateSync"] = num_particles
    # add a line for obs -- has to be size_obs in the input
    support_obs = 3
    connections["ObsStateSync", "WTA"] = num_particles + support_obs

    """ Randomly assign components to relevant brain regions """
    
    brain_regions = ["V1L2", "V1L4", "V1L5", "CP", "GPi", "LD"]
    if randomize_brain_regions:
        shuffle(brain_regions)
    snmc_components = ["WTA", "Assemblies", "Scoring", "Normalizer",
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
    return synapse_dictionary, compressed_synapse_dictionary, random_brain_snmc_pairings



def psps_real_vs_snmc(randomize_brain_regions):
    num_assemblies = 10
    assembly_size = 3
    num_particles = 1
    psp_ref = make_v1_psp_reference()
    psp_agnostic = psp_ref[-1]
    psp_excitatory = psp_ref[-2]
    components = ["WTA", "Assemblies", "Scoring"]
    layer_mappings = ["V1L2", "V1L4", "V1L5"]
    if randomize_brain_regions:
        shuffle(components)
    snmc_mappings = {l:c for (l, c) in zip(layer_mappings, components)}
    snmc_psps = component_internals(num_assemblies,
                                    assembly_size, snmc_mappings)[-1]
    compare_connectivity_heatmap([psp_agnostic, snmc_psps], ["Allen Micro Agnostic",
                                                             "SNMC Components"])
    compare_connectivity_scatter(psp_agnostic, snmc_psps)
    return snmc_psps
    # could make your cool loop figure idea here. i.e. if you place the micro components incorrectly, can't get the map. 
    
    




""" Macro Scale Connectivity """ 

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


def make_snmc_and_reference_dicts(real_df, assembly_size, num_assemblies,
                                  num_particles, syns_to_exclude):
    # normalize all values (i.e. total synapses)
    realsyn_dict = create_realsynapse_dictionary(real_df)
    for syn in syns_to_exclude:
        del realsyn_dict[syn]
    # these two references assign the brain regions according to the topology in the paper
    reference_snmc = generate_snmc_connectivity(
        assembly_size, num_assemblies, num_particles, False)[1]
    _, random_snmc, random_snmc_pairings = generate_snmc_connectivity(
        assembly_size, num_assemblies, num_particles, True)
    realkeys = realsyn_dict.keys()
    norm_syn_dicts = []
    all_syn_dicts = [realsyn_dict, reference_snmc, random_snmc]
    for d in all_syn_dicts:
        syn_dict = {k: d[k] for k in realkeys}
        total_synapses = np.sum(list(syn_dict.values()))
        syn_dict.update((k, v / total_synapses) for k, v in syn_dict.items())
        norm_syn_dicts.append(syn_dict)
    return norm_syn_dicts

        
def barplot_snmc_vs_real(norm_syn_dicts):
    network_labels = ["Real", "SNMC", "Shuffled"]
    realkeys = norm_syn_dicts[0].keys()
    x = [k[0]+"-"+k[1] for k in realkeys]
    y = [norm_syn_dicts[0][k] for k in realkeys]
    hue = np.repeat(network_labels[0], len(realkeys)).tolist()
    for k, v in norm_syn_dicts[0].items():
        print(k)
        x += np.repeat(k[0]+"-"+k[1], 2).tolist()
        y += [norm_syn_dicts[1][k], norm_syn_dicts[2][k]]
        hue += network_labels[1:]
    cpal = sns.color_palette("Set2")
    cpal = [cpal[0], cpal[2], cpal[3]]
    fig, ax = pl.subplots(1, 1, figsize=(12, 7))
    sns.barplot(x=x, y=y, hue=hue, palette=cpal, ax=ax)
    ax.set_xlabel("Synapse")
    ax.set_ylabel("Proportion of Circuit Connectivity")
    ax.tick_params(axis='x', labelsize=8, rotation=90)
    pl.show()


def average_n_snmc_simulations(n, syns_to_exclude, num_particles):
    rand_snmc_dicts = [make_snmc_and_reference_dicts(longrange, 3, 3, num_particles, syns_to_exclude)[-1]
                       for i in range(n)]
    avg_rand = {}
    for k, v in rand_snmc_dicts[0].items():
        avg_rand[k] = np.mean(list(map(lambda x: x[k], rand_snmc_dicts)))
    return avg_rand

# this is correct. distributes randomness evenly across all synapses 

def final_barplot_and_scatter(syns_to_exclude, num_particles):
    normalized_connectivity = make_snmc_and_reference_dicts(longrange, 3, 3, num_particles,
                                                            syns_to_exclude)
    avg_random_dict = average_n_snmc_simulations(1000, syns_to_exclude, num_particles)
    barplot_snmc_vs_real(normalized_connectivity + [avg_random_dict])
    make_scatter_from_syndicts(normalized_connectivity[0], normalized_connectivity[1], avg_random_dict)
    
def make_scatter_from_syndicts(real_d, comp_1, comp_2):
    cpal = sns.color_palette("Set2")
    xs = []
    ys_1 = []
    ys_2 = []
    yoffset = 0
    y_off = 0
    fig, ax = pl.subplots(1, 2)
    for k, v in real_d.items():
        x = v
        y1 = comp_1[k]
        if x == 0 and y1 == 0:
            y_off = copy.deepcopy(yoffset)
            yoffset += .02
        else:
            y_off = 0
        ax[0].annotate(k[0]+'-'+k[1], (x, y1+y_off))
        xs.append(x)
        ys_1.append(y1)
    yoffset = 0
    y_off = 0
    for k, v in real_d.items():
        x = v
        y2 = comp_2[k]
        if x == 0 and y2 == 0:
            y_off = copy.deepcopy(yoffset)
            yoffset += .02
        else:
            y_off = 0
        ax[1].annotate(k[0]+'-'+k[1], (x, y2+y_off))
        ys_2.append(y2)

    sns.scatterplot(x=xs, y=ys_1, ax=ax[0], color=cpal[3])
    sns.scatterplot(x=xs, y=ys_2, ax=ax[1], color=cpal[2])
#    pl.tight_layout()
    pl.show()
    
#     y_offset = 0
#     for xloc, yloc, syn in zip(x, y, synapses):
#         if xloc == 0 and yloc == 0:
#             y_off = copy.deepcopy(y_offset)
#             y_offset += .05
#             print(syn)
#         else:
#             y_off = 0
#         ax[1].annotate(syn, (xloc + .005, yloc + y_off))
#         ax[1].set_xlim([-.05, np.max(x) + .1])
#         ax[1].set_ylim([-.05, np.max(y) + .1])
#     pl.show()

syns_to_exclude = [("V1L5", "V1"), ("V1L2", "V1"), ("V1L4", "V1")]
snmc_dicts = make_snmc_and_reference_dicts(longrange, 3, 3, 1, syns_to_exclude)

interesting_random = {'V1L2': 'Assemblies',
                      'GPi': 'WTA',
                      'CP': 'Scoring',
                      'V1L5': 'Normalizer',
                      'V1L4': 'Resampler',
                      'LD': 'ObsStateSync',
                      'S1': 'DownstreamVariable'}
   

# def generate_snmc_psps(num_assemblies, assembly_size, snmc_type):
#     num_assemblies = 3
#     assembly_size = 3
#     snmc = generate_snmc_connectivity_bioreal(assembly_size, num_assemblies, 1, snmc_type, False)[-1]
#     random = generate_snmc_connectivity_bioreal(assembly_size, num_assemblies, 1, snmc_type, True)[-1]
#     return snmc, random


# def generate_snmc_connectivity_bioreal(assembly_size, num_assemblies, num_particles, snmc_type, randomize_brain_regions):

#     """ Synapses Per Neuron """

#     if snmc_type == "allen_micro":
#         e_wta_to_e_wta = 0
#         e_wta_to_i_wta = 1
#         i_wta_to_e_wta = 0
#         i_wta_to_i_wta = 0

#         e_wta_to_assembly_silencer = 1
#         i_wta_to_assembly_silencer = 1
        
#         e_wta_to_e_scoring = 0
#         e_wta_to_i_scoring = 0
#         i_wta_to_e_scoring = 2 # P mux and Q mux
#         i_wta_to_i_scoring = 0

#         e_wta_intracortical = 1 # state travels to next microcolumn.        

#         assembly_to_e_wta = 0
#         assembly_to_i_wta = 0
#         assembly_to_e_assembly_accumulator = 1
#         assembly_to_i_assembly_accumulator = 1
#         e_assembly_accumulator_to_e_wta = 1
#         e_assembly_accumulator_to_i_wta = 0
#         i_assembly_accumulator_to_e_wta = num_assemblies - 1
#         i_assembly_accumulator_to_i_wta = 0
#         assembly_silencer_to_e_assembly_accumulator = num_assemblies - 1
#         assembly_silencer_to_i_assembly_accumulator = num_assemblies - 1
#         assembly_to_e_scoring = 2
#         assembly_to_i_scoring = 0

#         q_thresholder_to_assembly_silencer = 1
#         p_thresholder_to_assembly_silencer = 1
#         q_accumulator_to_scoring = 1
#         q_accumulator_to_norm = 1
#         q_thresholder_to_scoring = 2
#         q_mux_to_scoring = 1
#         p_accumulator_to_scoring = 1
#         p_thresholder_to_scoring = 2
#         p_mux_to_norm = 1
#         inhibitory_mux_to_mux = 0  # one for each mux
#         inhibitory_scoring_neurons = 2
#         excitatory_scoring_neurons = 4

#         e_neurons_per_assembly = assembly_size + 1
#         i_neurons_per_assembly = 2
   
        
#     if snmc_type == "original":
#         # WTA
#         e_wta_to_e_wta = 0
#         e_wta_to_i_wta = 1
#         i_wta_to_e_wta = num_assemblies - 1
#         i_wta_to_i_wta = num_assemblies - 1

#         e_wta_to_assembly_silencer = 0
#         i_wta_to_assembly_silencer = 0
        
#         e_wta_to_e_scoring = 0
#         e_wta_to_i_scoring = 0
#         i_wta_to_e_scoring = 2
#         i_wta_to_i_scoring = 0
#         e_wta_intracortical = 1 # state travels to next microcolumn.
#         # Assemblies
#         assembly_to_e_wta = 1
#         assembly_to_i_wta = 0
#         assembly_to_e_scoring = 2
#         assembly_to_i_scoring = 0

#         assembly_to_e_assembly_accumulator = 0
#         assembly_to_i_assembly_accumulator = 0
#         e_assembly_accumulator_to_e_wta = 0
#         e_assembly_accumulator_to_i_wta = 0
#         i_assembly_accumulator_to_e_wta = 0
#         i_assembly_accumulator_to_i_wta = 0
#         assembly_silencer_to_e_assembly_accumulator = 0
#         assembly_silencer_to_i_assembly_accumulator = 0
#         e_neurons_per_assembly = assembly_size
#         i_neurons_per_assembly = 0
        
#         # Scoring
#         q_thresholder_to_assembly_silencer = 0
#         p_thresholder_to_assembly_silencer = 0        
#         q_accumulator_to_scoring = 1
#         q_accumulator_to_norm = 1
#         q_thresholder_to_scoring = 2
#         q_mux_to_scoring = 1
#         p_accumulator_to_scoring = 1
#         p_thresholder_to_scoring = 2
#         p_mux_to_norm = 1
#         inhibitory_mux_to_mux = 0  # one for each mux
#         inhibitory_scoring_neurons = 2
#         excitatory_scoring_neurons = 4

#         e_neurons_per_assembly = assembly_size
#         i_neurons_per_assembly = 0

        
#     if snmc_type == "update":
#         # WTA
#         e_wta_to_e_wta = 0
#         e_wta_to_i_wta = 0
#         i_wta_to_e_wta = num_assemblies - 1 # hits all other wta excitatory neurons
#         i_wta_to_i_wta = 0
#         e_wta_to_e_scoring = 0
#         e_wta_to_i_scoring = 1
#         i_wta_to_e_scoring = 0
#         i_wta_to_i_scoring = 0
#         e_wta_intracortical = 1 # state travels to next microcolumn.
#         e_wta_to_assembly_silencer = 0
#         i_wta_to_assembly_silencer = 0
#         # Assemblies
#         assembly_to_e_wta = 1
#         assembly_to_i_wta = 1
#         assembly_to_e_scoring = 2
#         assembly_to_i_scoring = 0

#         assembly_to_e_assembly_accumulator = 0
#         assembly_to_i_assembly_accumulator = 0
#         e_assembly_accumulator_to_e_wta = 0
#         e_assembly_accumulator_to_i_wta = 0
#         i_assembly_accumulator_to_e_wta = 0
#         i_assembly_accumulator_to_i_wta = 0
#         assembly_silencer_to_e_assembly_accumulator = 0
#         assembly_silencer_to_i_assembly_accumulator = 0
#         e_neurons_per_assembly = assembly_size
#         i_neurons_per_assembly = 0
#         # Scoring
#         q_thresholder_to_assembly_silencer = 0
#         p_thresholder_to_assembly_silencer = 0
#         q_accumulator_to_scoring = 1
#         q_accumulator_to_norm = 1
#         q_thresholder_to_scoring = 2
#         q_mux_to_scoring = 1
#         p_accumulator_to_scoring = 1
#         p_thresholder_to_scoring = 2
#         p_mux_to_norm = 1
#         inhibitory_mux_to_mux = 2  # one for each mux
#         inhibitory_scoring_neurons = num_assemblies + 2
#         excitatory_scoring_neurons = 4
        
#     excitatory_wta_to_wta = e_wta_to_e_wta + e_wta_to_i_wta
#     inhibitory_wta_to_wta = i_wta_to_e_wta + i_wta_to_i_wta
#     excitatory_wta_to_scoring = e_wta_to_e_scoring + e_wta_to_i_scoring
#     inhibitory_wta_to_scoring = i_wta_to_e_scoring + i_wta_to_i_scoring
#     assembly_to_wta = assembly_to_e_wta + assembly_to_i_wta
#     assembly_to_scoring = assembly_to_e_scoring + assembly_to_i_scoring

#     psp_probs = {}
#     psp_probs["WTA_e", "WTA_e"] = e_wta_to_e_wta / (num_assemblies-1)
#     psp_probs["WTA_e", "WTA_i"] = e_wta_to_i_wta / num_assemblies
#     psp_probs["WTA_i", "WTA_e"] = i_wta_to_e_wta / num_assemblies
#     psp_probs["WTA_i", "WTA_i"] = i_wta_to_i_wta / (num_assemblies-1)
    
#     psp_probs["WTA_e", "Assemblies_e"] = 0
#     psp_probs["WTA_i", "Assemblies_e"] = 0
    
#     psp_probs["WTA_e", "Assemblies_i"] = e_wta_to_assembly_silencer / num_assemblies
#     psp_probs["WTA_i", "Assemblies_i"] = 0
    
#     psp_probs["WTA_e", "Scoring_i"] = e_wta_to_i_scoring / inhibitory_scoring_neurons
#     psp_probs["WTA_i", "Scoring_i"] = i_wta_to_i_scoring / inhibitory_scoring_neurons
#     psp_probs["WTA_e", "Scoring_e"] = e_wta_to_e_scoring / excitatory_scoring_neurons
#     psp_probs["WTA_i", "Scoring_e"] = i_wta_to_e_scoring / excitatory_scoring_neurons

    
    
#     psp_probs["Assemblies_e", "WTA_e"] = (assembly_size / e_neurons_per_assembly) * assembly_to_e_wta / num_assemblies + (1 / e_neurons_per_assembly) * e_assembly_accumulator_to_e_wta / num_assemblies 
#     psp_probs["Assemblies_i", "WTA_e"] = .5 * i_assembly_accumulator_to_e_wta / num_assemblies 
#     psp_probs["Assemblies_e", "WTA_i"] = assembly_to_i_wta / num_assemblies
#     psp_probs["Assemblies_i", "WTA_i"] = 0


#     # this isn't correct. 
#     psp_probs["Assemblies_e", "Assemblies_e"] = (assembly_size / e_neurons_per_assembly) * ((e_neurons_per_assembly-assembly_size) / (num_assemblies * e_neurons_per_assembly))

    
#     psp_probs["Assemblies_e", "Assemblies_i"] = (assembly_size / e_neurons_per_assembly) * (assembly_to_i_assembly_accumulator / (num_assemblies * i_neurons_per_assembly))
     
#     psp_probs["Assemblies_i", "Assemblies_e"] = (assembly_silencer_to_e_assembly_accumulator / i_neurons_per_assembly) * (e_neurons_per_assembly - assembly_size) / e_neurons_per_assembly
    
#     psp_probs["Assemblies_i", "Assemblies_i"] = (assembly_silencer_to_i_assembly_accumulator / i_neurons_per_assembly) * .5 # connects to all inhib out to wta, no silencers. make this more formal there's no current description the amount of inhibitory output neurons. 

    
#     psp_probs["Assemblies_e", "Scoring_e"] = (assembly_size / e_neurons_per_assembly) * assembly_to_e_scoring / excitatory_scoring_neurons
#     psp_probs["Assemblies_i", "Scoring_e"] = 0
#     psp_probs["Assemblies_e", "Scoring_i"] = (assembly_size / e_neurons_per_assembly) * assembly_to_i_scoring / inhibitory_scoring_neurons
#     psp_probs["Assemblies_i", "Scoring_i"] = 0

     
#     psp_probs["Scoring_e", "WTA_e"] = 0
#     psp_probs["Scoring_e", "WTA_i"] = 0
#     psp_probs["Scoring_i", "WTA_i"] = 0
#     psp_probs["Scoring_i", "WTA_e"] = 0
#     psp_probs["Scoring_e", "Assemblies_e"] = 0

#     psp_probs["Scoring_e", "Assemblies_i"] = 0

#     psp_probs["Scoring_i", "Assemblies_e"] = 0
#      # every inhibitory neuron projects to every silencer, no L4 output neurons. 
#     psp_probs["Scoring_i", "Assemblies_i"] = .5
     
#     psp_probs["Scoring_e", "Scoring_i"] = (.25 * q_mux_to_scoring / inhibitory_scoring_neurons) + (
#         .25 * p_accumulator_to_scoring / inhibitory_scoring_neurons) # two other excitatory types pmux to norm and q_accum to norm, which don't connect to anything locally.
#     psp_probs["Scoring_i", "Scoring_i"] = 0
#     psp_probs["Scoring_e", "Scoring_e"] = 0
#     psp_probs["Scoring_i", "Scoring_e"] = (1 / inhibitory_scoring_neurons) * q_thresholder_to_scoring / excitatory_scoring_neurons + (1 / inhibitory_scoring_neurons) * p_thresholder_to_scoring / excitatory_scoring_neurons + num_assemblies / inhibitory_scoring_neurons * (inhibitory_mux_to_mux / excitatory_scoring_neurons) 
#     # probabilities here are more complex because there are multiple excitatory neurons that each have a different
#     # connection probability.
#     microcircuit_layers = ["L2", "L4", "L5"]
#     microcircuit_components = ["WTA", "Assemblies", "Scoring"]
#     if randomize_brain_regions:
#         shuffle(microcircuit_components)
#     psp_df = pd.DataFrame(columns=list(map(
#         lambda l: l + "e", microcircuit_layers)) + list(
#             map(lambda l: l + "i", microcircuit_layers)))
#     layers_ei = psp_df.columns
#     psp_df["PreSyn"] = layers_ei
#     random_microcircuit_pairings = {a: b for a, b in zip(
#         psp_df.columns, list(map(
#             lambda l: l + "_e", microcircuit_components)) + list(
#                 map(lambda l: l + "_i", microcircuit_components)))}
    
#     for syn in itertools.product(layers_ei, layers_ei):
#         psp_df.loc[psp_df["PreSyn"] == syn[0], syn[1]] = psp_probs[random_microcircuit_pairings[syn[0]],
#                                                                    random_microcircuit_pairings[syn[1]]]
        
#     """ Build a dictionary that establishes a mapping between components """
        
#     connections = {}
#     connections["Scoring", "Scoring"] = q_accumulator_to_scoring + q_accumulator_to_norm + q_thresholder_to_scoring + q_mux_to_scoring + p_accumulator_to_scoring + p_thresholder_to_scoring + p_mux_to_norm + (num_assemblies * inhibitory_mux_to_mux)
#     connections["WTA", "WTA"] = num_assemblies * (excitatory_wta_to_wta + inhibitory_wta_to_wta)
#     connections["WTA", "Scoring"] = num_assemblies * (inhibitory_wta_to_scoring + excitatory_wta_to_scoring)
#     connections["WTA", "DownstreamVariable"] = num_assemblies * e_wta_intracortical
#     connections["Assemblies", "WTA"] = num_assemblies * assembly_size * assembly_to_wta
#     connections["Assemblies", "Scoring"] = num_assemblies * assembly_size * assembly_to_scoring
#     connections.update((k, v*num_particles) for k, v in connections.items())
    
#     connections["Scoring", "Normalizer"] = 2 * num_particles
#     connections["Normalizer", "Resampler"] = num_particles
#     connections["Resampler", "ObsStateSync"] = num_particles
#     connections["ObsStateSync", "WTA"] = num_particles

#     """ Randomly assign components to relevant brain regions """
    
#     brain_regions = ["V1L2", "V1L4", "V1L5", "CP", "GPi", "LD"]
#     if randomize_brain_regions:
#         shuffle(brain_regions)
        
#     snmc_components = ["WTA", "Assemblies", "Scoring", "Normalizer",
#                        "Resampler", "ObsStateSync"]
#     random_brain_snmc_pairings = dict(zip(brain_regions, snmc_components))
#     random_brain_snmc_pairings["S1"] = "DownstreamVariable"
#     pairwise_synapses = [(a[0], a[1]) for a in list(itertools.product(brain_regions + ["S1"], brain_regions + ["S1"]))]

#     """ Project the component mapping onto the random brain regions """ 
    
#     def map_brain_to_snmc(source, termination):
#         try:
#             source_brain = random_brain_snmc_pairings[source]
#             termination_brain = random_brain_snmc_pairings[termination]
#             return connections[source_brain, termination_brain]
#         except KeyError:
#             return 0

#     synapse_dictionary = {}
#     for pw_syn in pairwise_synapses:
#         synapse_dictionary[pw_syn[0], pw_syn[1]] = map_brain_to_snmc(pw_syn[0], pw_syn[1])

#     """ Compress the resolution of the random mapping to the figure """

#     compressed_synapse_dictionary = {}
#     cp_to_v1_accumulator = 0
#     gpi_to_v1_accumulator = 0
#     thal_to_v1_accumulator = 0
#     v1l5_to_v1_accumulator = 0
#     v1l4_to_v1_accumulator = 0
#     v1l2_to_v1_accumulator = 0
    
#     for k, v in synapse_dictionary.items():
#         if k[0] == "CP" and k[1][0:2] == "V1":
#             cp_to_v1_accumulator += v
#         elif k[0] == "GPi" and k[1][0:2] == "V1":
#             gpi_to_v1_accumulator += v
#         elif k[0] == "LD" and k[1][0:2] == "V1":
#             thal_to_v1_accumulator += v
#         elif k[0] == "V1L5" and k[1][0:2] == "V1":
#             v1l5_to_v1_accumulator += v
#         elif k[0] == "V1L4" and k[1][0:2] == "V1":
#             v1l4_to_v1_accumulator += v
#         elif k[0] == "V1L2" and k[1][0:2] == "V1":
#             v1l2_to_v1_accumulator += v
#         else:
#             compressed_synapse_dictionary[k] = v

#     compressed_synapse_dictionary["CP", "V1"] = cp_to_v1_accumulator
#     compressed_synapse_dictionary["GPi", "V1"] = gpi_to_v1_accumulator
#     compressed_synapse_dictionary["LD", "V1"] = thal_to_v1_accumulator
#     compressed_synapse_dictionary["V1L5", "V1"] = v1l5_to_v1_accumulator
#     compressed_synapse_dictionary["V1L4", "V1"] = v1l4_to_v1_accumulator
#     compressed_synapse_dictionary["V1L2", "V1"] = v1l2_to_v1_accumulator
# #     return synapse_dictionary, compressed_synapse_dictionary, random_brain_snmc_pairings, psp_df
# def component_pmap():
#     layers = ["L2", "L4", "L5"]
#     df = pd.DataFrame(columns=layers+["PreSyn"])
#     df["PreSyn"] = layers
#     df.loc[df["PreSyn"] == "L2", "L2"] = 0
#     df.loc[df["PreSyn"] == "L2", "L4"] = 0
#     df.loc[df["PreSyn"] == "L2", "L5"] = 2/8 # WTA hits only muxes
#     df.loc[df["PreSyn"] == "L4", "L2"] = .5 # half of assemblies hit WTA w/ 100% prob
#     df.loc[df["PreSyn"] == "L4", "L4"] = 0  # assemblies aren't interconnected
#     df.loc[df["PreSyn"] == "L4", "L5"] = 2/8 # assemblies hit one mux and one accum.
#     df.loc[df["PreSyn"] == "L4", "L5"] = 2/8 # assemblies hit one mux and one accum.
#     df.loc[df["PreSyn"] == "L5", "L2"] = 0  # scoring doesn't talk to WTA_e
#     df.loc[df["PreSyn"] == "L5", "L4"] = 0  # scoring doesn't project back to assemblie
#     df.loc[df["PreSyn"] == "L5", "L5"] = 6/64
#     # For last calculation: 
#     # q accum only out: (1/8) * 0
#     # q mux to integrator: (1/8) * (1/8)
#     # both integrators to thresholders: (1/4) * (1/8)
#     # qthresh to qaccum: (1/8) * (1/8)
#     # pthresh to pmux: (1/8) * (1/8)
#     # paccum to pint: (1/8) * (1/8)
#     # pmux to nothing: (1/8) * 0
#     return df
