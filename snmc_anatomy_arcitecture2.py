import numpy as np
import seaborn as sns
import pandas as pd
from functools import reduce
from matplotlib import pyplot as pl
import matplotlib.patches as mpatches
import copy
from random import shuffle
import itertools
import math
import fractions
import random
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_squared_error
from more_itertools import chunked


# difference between architecture 2 and original is WTA reports state to the assemblies
# via a connection to a single neuron per assembly per state. no more all to all, which doesn't appear to be realistic. 
# also there is no intermediate MUX neuron. there is only one per state where the layer 4
# neurons all go to. it doesn't go single neuron then converge. that makes no sense. this gets
# rid of the black nodes (which i created as neurons). if you want to go back you can by putting the MUX size back to snmc_anatomy.py
# and readjusting the MUX - MUX and Assembly to MUX probs.

# i also added a connection to stop the state reporting neurons in L4 from firing after TIK hits its count.

longrange = pd.read_csv("longrange.csv")
longrange_bar = pd.read_csv("longrange_df.csv")
microcircuit = pd.read_csv("microcircuit.csv")
microcircuit_probability = pd.read_csv("microcircuit_probabilities.csv")
microcircuit_probability_bar = pd.read_csv("microcircuit_probs_df.csv")


# components are WTA, MUX, Gate, TIK, Assemblies. what you need is a total neuron count for each component which adds to a total neuron count for a layer.
# each component has a probability of connecting to another component's neurons. the total number in any given layer is the sum of the components assigned to the layer.

# e.g. (num_assemblies * assembly_size) * 1 / total_layer_containing_gate. each function will have to access a dictionary about where the other structures are and
# the consequent total number of neurons per layer. write a function that takes the dictionary and returns a total number of neurons for the arg layer.


# you want to make sure that you have a better abstraction for dataframes. its totally unwieldy to always type in the loc syntax. make a class that wraps a dataframe and write an "assign" and "value" function for it so you don't have to keep
# indexing and assigning with the unwieldy syntax. 

def total_layer_neurons(component, layer_assignment, component_counts):
    layer = layer_assignment[component]
    total_neurons_in_layer = reduce(lambda x, y: x + y, (map(
        lambda x: component_counts[x] if layer_assignment[x] == layer else 0,
        layer_assignment.keys())))
    return total_neurons_in_layer

def assign_components_to_layers(num_assemblies, assembly_size):
    assembly_autonorm_size = 3
    component_counts = {"MUX": (2 * num_assemblies),
                        "GATE": 2,
                        "TIK": 2,
                        "WTA": num_assemblies * 3,
                        "Assemblies": (num_assemblies * num_assemblies) + assembly_autonorm_size + (
                            num_assemblies * assembly_size)}
    component_counts_pq = {k: component_counts[k[0]] for k in
                           itertools.product(component_counts.keys(),
                                             ["P", "Q"])}
    del component_counts_pq[("WTA", "P")]
    assignments = {c: np.random.choice(["L2", "L4", "L5"]) for c in component_counts_pq.keys()}
    return assignments, component_counts_pq



def calculate_connection_probabilities(layer_assignment, component_counts, num_assemblies, assembly_size, assembly_autonorm_size):
    # these are going to be layer by layer.
    # first make a dataframe with pre and post by layer
    # iterate through the product of layers 2,4,5.
    # collect all components in pre and post.
    # calculate probabilities based on total neurons in each layer.
    # "total neurons" returns the total size of the layer that the component is assigned to.
# call is probability_table(k[0][0], k[1][0])(k[0][1], k[1][1]) after iterating through product of all elements with p and q. 
# now you'll get a connection probability for each pairwise p and q component in terms of layers.
    components = ["MUX", "GATE", "TIK", "WTA", "Assemblies"]
    # initialize w zero prob connections
    probability_table = {c: lambda p: 0 for c in itertools.product(components,
                                                                   components)}
    ly = layer_assignment
    cc = component_counts
    # WTA theta gate connects to all wta blue. wta blue connect to only one red. red connect to only one blue.
    # there are num_assemblies of each.
    probability_table["WTA", "WTA"] = lambda pq: (num_assemblies / total_layer_neurons(("WTA", "Q"), ly, cc)) * (num_assemblies / total_layer_neurons(("WTA", "Q"), ly, cc)) + 2 * (num_assemblies / total_layer_neurons(("WTA", "Q"), ly, cc)) * (1 / total_layer_neurons(("WTA", "Q"), ly, cc))

    # this should be num_assemblies * assembly_size --- its one to all fan out.
    probability_table["WTA", "Assemblies"] = lambda pq: (num_assemblies / total_layer_neurons(("WTA", pq[0]), ly, cc)) * (num_assemblies / total_layer_neurons(("Assemblies", pq[1]), ly, cc))
    
    probability_table["WTA", "MUX"] = lambda pq: (num_assemblies / total_layer_neurons(("WTA", pq[0]), ly, cc)) * (1 / total_layer_neurons(("MUX", pq[1]), ly, cc))

    probability_table["Assemblies", "WTA"] = lambda pq: ((num_assemblies * assembly_size) / total_layer_neurons(
        ("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(("WTA", "Q"), ly, cc))

    probability_table["Assemblies", "Assemblies"] = lambda pq: 0 if pq[0] != pq[1] else (num_assemblies * assembly_size / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (2 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) + (2 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (num_assemblies * assembly_size / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) + (1 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) + (num_assemblies * num_assemblies / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (assembly_size / total_layer_neurons(("Assemblies", pq[0]), ly, cc))
                                                 
    probability_table["Assemblies", "MUX"] = lambda pq: 0 if pq[0] != pq[1] else ((num_assemblies * assembly_size) / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(("MUX", pq[1]), ly, cc))

    probability_table["Assemblies", "GATE"] = lambda pq: ((num_assemblies * assembly_size) / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(("GATE", pq[1]), ly, cc)) if pq[0] == pq[1] == "Q" else 0

    probability_table["Assemblies", "TIK"] = lambda pq: ((num_assemblies * assembly_size) / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(("TIK", pq[1]), ly, cc)) if pq[0] == pq[1] == "P" else 0

    probability_table["MUX", "TIK"] = lambda pq: (num_assemblies / total_layer_neurons(("MUX", pq[0]), ly, cc)) * (1 / total_layer_neurons(("TIK", pq[1]), ly, cc)) if pq[0] == pq[1] == "Q" else 0

    probability_table["MUX", "GATE"] = lambda pq: (num_assemblies / total_layer_neurons(("MUX", pq[0]), ly, cc)) * (1 / total_layer_neurons(("GATE", pq[1]), ly, cc)) if pq[0] == pq[1] == "P" else 0

    probability_table["TIK", "WTA"] = lambda pq: (1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (num_assemblies / total_layer_neurons(("WTA", pq[1]), ly, cc))

    probability_table["TIK", "Assemblies"] = lambda pq: (1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * ((assembly_autonorm_size - 1) / total_layer_neurons(("Assemblies", pq[1]), ly, cc)) + (1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (num_assemblies * num_assemblies) / total_layer_neurons(("Assemblies", pq[1]), ly, cc)

    probability_table["TIK", "GATE"] = lambda pq: 0 if pq[0] != pq[1] else (1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (1 / total_layer_neurons(("GATE", pq[1]), ly, cc))

    probability_table["TIK", "TIK"] = lambda pq: 0 if pq[0] != pq[1] else (1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (1 / total_layer_neurons(("TIK", pq[1]), ly, cc))

    # blue to red and red to blue. each blue to one red each red to one blue.
    probability_table["MUX", "MUX"] = lambda pq: 0 if pq[0] != pq[1] else 2 * (num_assemblies / total_layer_neurons(("MUX", pq[0]), ly, cc)) * (1 / total_layer_neurons(("MUX", pq[1]), ly, cc))
    layers = ["L2", "L4", "L5"]
    layer_df = pd.DataFrame(np.zeros(shape=(len(layers), len(layers))), columns=layers)
    layer_df["PreSyn"] = layers
    for pre, post in itertools.product(layer_assignment.keys(), layer_assignment.keys()):
        layer_df.loc[
            layer_df["PreSyn"] == layer_assignment[pre], layer_assignment[post]] += probability_table[pre[0], post[0]]([pre[1], post[1]])
    return probability_table, layer_df



# run this a bunch of times -- compare to a groundtruth dataset. use seaborn error plots. 


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

        all_connections = reduce(
            lambda a, b: a + b, [term_filtered[c].tolist() for c in term_filtered.columns])
        all_connections = list(filter(lambda a: type(a) == fractions.Fraction, all_connections))
        if len(all_connections) != 0:
            frac_connection = reduce(
                lambda a, b: fractions.Fraction(
                    a.numerator + b.numerator, a.denominator + b.denominator, _normalize=False),
                all_connections)
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
    excitatory_weight = .8
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


# here want to redo this in component space. make a random dictionary that goes through each iter.product of layers and indexes the
# assignment dictionary. 



def random_connectivity_errorplot(num_random_sims, assembly_size, num_assemblies):
    psp_ref = make_v1_psp_reference()
    psp_agnostic = psp_ref[-1]
    psp_excitatory = psp_ref[-2]
    autonorm_size = 3
    random_dfs = []
    random_assignments = []
    for s in range(num_random_sims):
        l_assgn, compcounts = assign_components_to_layers(num_assemblies, assembly_size)
        probtable, random_connections = calculate_connection_probabilities(
            l_assgn, compcounts, num_assemblies, assembly_size, autonorm_size)
        random_dfs.append(random_connections)
        random_assignments.append(l_assgn)
        
    real_values = []
    prob_values = []
    layer_combos = list(itertools.product(["L2", "L4", "L5"], ["L2", "L4", "L5"]))
    regression_fits = []
    for rand_df in random_dfs:
        real_for_df = []
        prob_for_df = []
        for pre, post in layer_combos:
            real_for_df.append(np.round(psp_agnostic.loc[psp_agnostic["PreSyn"] == pre, post].values[0], 4))
            prob_for_df.append(np.round(rand_df.loc[rand_df["PreSyn"] == pre, post].values[0], 4))
        slope, intercept, r_value, p_value, std_err = linregress(real_for_df, prob_for_df)
        regression_fits.append((slope, r_value))
        real_values += real_for_df
        prob_values += prob_for_df
    
    shepard_informed_assignments = {('MUX', 'P'): 'L5',
                                    ('MUX', 'Q'): 'L5',
                                    ('GATE', 'P'): 'L5',
                                    ('GATE', 'Q'): 'L5',
                                    ('TIK', 'P'): 'L5',
                                    ('TIK', 'Q'): 'L5',
                                    ('WTA', 'Q'): 'L2',
                                    ('Assemblies', 'P'): 'L4',
                                    ('Assemblies', 'Q'): 'L4'}
    
    our_snmc_df = calculate_connection_probabilities(shepard_informed_assignments,
                                                     compcounts,
                                                     num_assemblies,
                                                     assembly_size,
                                                     autonorm_size)[1]
    snmc_probs = []
    for pre, post in layer_combos:
        snmc_probs.append(np.round(
            our_snmc_df.loc[our_snmc_df["PreSyn"] == pre, post].values[0], 4))
        

    snmc_slope, intercept, snmc_r_value, p_value, std_err = linregress(real_for_df, snmc_probs)
    regression_fits.append((snmc_slope, snmc_r_value))
    random_assignments.append(shepard_informed_assignments)
    avg_random_slope, intercept, avg_random_r_value, p_value, std_err = linregress(real_for_df, np.mean(list(chunked(prob_values, len(layer_combos))), axis=0))
    print(snmc_slope, snmc_r_value)
    print(avg_random_slope, avg_random_r_value)
    fig, ax = pl.subplots(1, 1)
    cpal = sns.color_palette("Set2")
    sns.pointplot(x=real_values, y=prob_values, estimator=np.median, ci=95, join=False, color=cpal[1], ax=ax)
    sns.pointplot(x=real_values[0:len(layer_combos)], y=snmc_probs, join=False, color=cpal[2], ax=ax)
    sns.pointplot(x=real_values, y=real_values, markers='', color=cpal[0], ax=ax)
    sorted_realvals = np.sort(real_values[0:len(layer_combos)])
    argsorted_realvals = np.argsort(real_values[0:len(layer_combos)])
    for i, (rv, rvi) in enumerate(zip(sorted_realvals, argsorted_realvals)):
        ax.annotate(layer_combos[rvi], (i, rv+.01))

    c0 = mpatches.Patch(color=cpal[0], label='Real')
    c1 = mpatches.Patch(color=cpal[1], label='Random Component Assignment')
    c2 = mpatches.Patch(color=cpal[2], label='Our Arrangement')
    ax.set_xlabel("Real")
    ax.set_ylabel("SNMC")
    pl.legend(handles=[c0, c1, c2])
    pl.show()

    mse_slope = [np.sqrt((s[0] - 1)**2) for s in regression_fits]
    mse_corr = [np.sqrt((r[1] - 1)**2) for r in regression_fits]
    closest_fits = np.argsort([s+c for s, c in zip(mse_slope, mse_corr)])
    rank_of_our_snmc = np.where(closest_fits == num_random_sims)
    
    return regression_fits, random_assignments, closest_fits, rank_of_our_snmc

  

def compare_connectivity_scatter(real_df, snmc_df, plotit):
    if plotit:
        fig, ax = pl.subplots(1, 1)
    xvals = []
    yvals = []
    for pre, post in itertools.product(["L2", "L4", "L5"], ["L2", "L4", "L5"]):
        xval = real_df.loc[real_df["PreSyn"] == pre, post].values[0]
        yval = snmc_df.loc[snmc_df["PreSyn"] == pre, post].values[0]
        xvals.append(float(xval))
        yvals.append(float(yval))
        if plotit:
            ax.annotate(pre+"-" + post, (xval+.001, yval))
    cpal = sns.color_palette("Set2")
    if plotit:
        ax.scatter(x=xvals, y=yvals, color=cpal[3])
        ax.plot(range(5), range(5))
        ax.set_xlabel("Real")
        ax.set_ylabel("SNMC")
        ax.set_xlim([0, .2])
        ax.set_ylim([0, .5])
        pl.tight_layout()
        pl.show()
    slope, intercept, r_value, p_value, std_err = linregress(xvals, yvals)
    return slope, r_value

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



def fit_params_to_connectivity():
    div = 5
    num_assemblies = range(3, 20, div)
    assembly_sizes = range(3, 100, div)
    psp_ref = make_v1_psp_reference()
    psp_agnostic = psp_ref[-1]
    components = ["WTA", "Assemblies", "Scoring"]
    layer_mappings = ["V1L2", "V1L4", "V1L5"]
    r_vals = []
    r_vals_random = []
    s_vals = []
    s_vals_random = []
    snmc_mappings = {l:c for (l, c) in zip(layer_mappings, components)}
    shuffle(components)
    snmc_mappings_random = {l:c for (l, c) in zip(layer_mappings, components)}
    x_3d = []
    y_3d = []
    for na in num_assemblies:
        for ass_size in assembly_sizes:
            snmc_psps = component_internals(na,
                                            ass_size, snmc_mappings)[-1]
            
            s, r = compare_connectivity_scatter(psp_agnostic, snmc_psps, False)
            r_vals.append(r)
            s_vals.append(s)
            snmc_psps_rand = component_internals(na,
                                                 ass_size, snmc_mappings_random)[-1]
            s_rand, r_rand = compare_connectivity_scatter(psp_agnostic, snmc_psps_rand, False)
            r_vals_random.append(r_rand)
            s_vals_random.append(s_rand)
            x_3d.append(na)
            y_3d.append(ass_size)
    fig = pl.figure(figsize=(8, 8))
    cpal = sns.color_palette("Set2")
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax1.scatter(np.array(x_3d), np.array(y_3d), zs=np.array(r_vals), color=cpal[1])
    ax2.scatter(np.array(x_3d), np.array(y_3d), zs=np.array(r_vals_random), color=cpal[3])
    ax3.scatter(np.array(x_3d), np.array(y_3d), zs=np.array(s_vals), color=cpal[1])
    ax4.scatter(np.array(x_3d), np.array(y_3d), zs=np.array(s_vals_random), color=cpal[3])
    ax1.set_xlabel("Number of Assemblies")
    ax1.set_ylabel("Assembly Size")
    ax2.set_xlabel("Number of Assemblies")
    ax2.set_ylabel("Assembly Size")
    ax1.set_title("SNMC - Real Correlation")
    ax2.set_title("SNMC Random - Real Correlation")
    ax3.set_title("SNMC - Real Slope")
    ax4.set_title("SNMC Random - Real Slope")
    ax1.set_xlabel("Number of Assemblies")
    ax1.set_ylabel("Assembly Size")
    ax2.set_xlabel("Number of Assemblies")
    ax2.set_ylabel("Assembly Size")
    ax3.set_xlabel("Number of Assemblies")
    ax3.set_ylabel("Assembly Size")
    ax4.set_xlabel("Number of Assemblies")
    ax4.set_ylabel("Assembly Size")

    print(components)
    pl.show()





    
    


def psps_real_vs_snmc(psp_params, randomize_brain_regions):
    num_assemblies, assembly_size = psp_params
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
    r = compare_connectivity_scatter(psp_agnostic, snmc_psps, True)
    print(snmc_mappings)
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
    reference_snmc = generate_snmc_macro_connectivity(
        assembly_size, num_assemblies, num_particles, False)[1]
    _, random_snmc, random_snmc_pairings = generate_snmc_macro_connectivity(
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
   

