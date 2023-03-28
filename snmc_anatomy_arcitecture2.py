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
#                        "WTA_Dep": num_assemblies * 3,
                        "Assemblies": (num_assemblies * num_assemblies) + assembly_autonorm_size + (
                            num_assemblies * assembly_size)}
    component_counts_pq = {k: component_counts[k[0]] for k in
                           itertools.product(component_counts.keys(),
                                             ["P", "Q"])}
    del component_counts_pq[("WTA", "P")]
 #   del component_counts_pq[("WTA_Dep", "P")]
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
#    components = ["MUX", "GATE", "TIK", "WTA", "WTA_Dep", "Assemblies"]
    components = ["MUX", "GATE", "TIK", "WTA", "Assemblies"]
    # initialize w zero prob connections
    probability_table = {c: lambda p: 0 for c in itertools.product(components,
                                                                   components)}
    ly = layer_assignment
    cc = component_counts
    # WTA theta gate connects to all wta blue. wta blue connect to only one red. red connect to only one blue.
    # there are num_assemblies of each.
    probability_table["WTA", "WTA"] = lambda pq: (
        num_assemblies / total_layer_neurons(("WTA", "Q"), ly, cc)) * (
        (num_assemblies - 1) / total_layer_neurons(("WTA", "Q"), ly, cc)) + 2 * (
            num_assemblies / total_layer_neurons(
                ("WTA", "Q"), ly, cc)) * (1 / total_layer_neurons(("WTA", "Q"), ly, cc))
    # this should be num_assemblies * assembly_size --- its one to all fan out.
    probability_table["WTA", "Assemblies"] = lambda pq: (
        num_assemblies / total_layer_neurons(("WTA", pq[0]), ly, cc)) * (
            num_assemblies / total_layer_neurons(("Assemblies", pq[1]), ly, cc))
    probability_table["WTA", "MUX"] = lambda pq: (
        num_assemblies / total_layer_neurons(
            ("WTA", pq[0]), ly, cc)) * (1 / total_layer_neurons(("MUX", pq[1]), ly, cc))
    probability_table["Assemblies", "WTA"] = lambda pq: (
        (num_assemblies * assembly_size) / total_layer_neurons(
            ("Assemblies", pq[0]), ly, cc)) * (2 / total_layer_neurons(
                ("WTA", pq[1]), ly, cc)) if pq[0] == pq[1] == "Q" else 0
    probability_table["Assemblies", "Assemblies"] = lambda pq: 0 if pq[0] != pq[1] else (num_assemblies * assembly_size / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (2 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) + (2 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (num_assemblies * assembly_size / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) + (1 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) + (num_assemblies * num_assemblies / total_layer_neurons(("Assemblies", pq[0]), ly, cc)) * (assembly_size / total_layer_neurons(("Assemblies", pq[0]), ly, cc))
    probability_table["Assemblies", "MUX"] = lambda pq: 0 if pq[0] != pq[1] else (
        (num_assemblies * assembly_size) / total_layer_neurons(
            ("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(("MUX", pq[1]), ly, cc))
    probability_table["Assemblies", "GATE"] = lambda pq: (
        (num_assemblies * assembly_size) / total_layer_neurons(
            ("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(
                ("GATE", pq[1]), ly, cc)) if pq[0] == pq[1] == "Q" else 0
    probability_table["Assemblies", "TIK"] = lambda pq: (
        (num_assemblies * assembly_size) / total_layer_neurons(
            ("Assemblies", pq[0]), ly, cc)) * (1 / total_layer_neurons(
                ("TIK", pq[1]), ly, cc)) if pq[0] == pq[1] == "P" else 0
    probability_table["MUX", "TIK"] = lambda pq: (
        num_assemblies / total_layer_neurons(("MUX", pq[0]), ly, cc)) * (
            1 / total_layer_neurons(("TIK", pq[1]), ly, cc)) if pq[0] == pq[1] == "Q" else 0
    probability_table["MUX", "GATE"] = lambda pq: (
        num_assemblies / total_layer_neurons(("MUX", pq[0]), ly, cc)) * (
            1 / total_layer_neurons(("GATE", pq[1]), ly, cc)) if pq[0] == pq[1] == "P" else 0
    probability_table["TIK", "WTA"] = lambda pq: (1 / total_layer_neurons(
        ("TIK", pq[0]), ly, cc)) * (num_assemblies / total_layer_neurons(("WTA", pq[1]), ly, cc))
    probability_table["TIK", "Assemblies"] = lambda pq: (
        1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (
            (num_assemblies * assembly_size) / total_layer_neurons(
                ("Assemblies", pq[1]), ly, cc)) + (1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (
                    num_assemblies * num_assemblies) / total_layer_neurons(("Assemblies", pq[1]), ly, cc) if pq[0] == pq[1] else 0
    probability_table["TIK", "GATE"] = lambda pq: 0 if pq[0] != pq[1] else (
        1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (1 / total_layer_neurons(("GATE", pq[1]), ly, cc))
    probability_table["TIK", "TIK"] = lambda pq: 0 if pq[0] != pq[1] else (
        1 / total_layer_neurons(("TIK", pq[0]), ly, cc)) * (1 / total_layer_neurons(("TIK", pq[1]), ly, cc))
    # blue to red and red to blue. each blue to one red each red to one blue.
    probability_table["MUX", "MUX"] = lambda pq: 0 if pq[0] != pq[1] else 2 * (
        num_assemblies / total_layer_neurons(("MUX", pq[0]), ly, cc)) * (
            1 / total_layer_neurons(("MUX", pq[1]), ly, cc))
    layers = ["L2", "L4", "L5"]
    layer_df = pd.DataFrame(np.zeros(shape=(
        len(layers), len(layers))), columns=layers)
    layer_df["PreSyn"] = layers
    for pre, post in itertools.product(layer_assignment.keys(), layer_assignment.keys()):
        layer_df.loc[
            layer_df["PreSyn"] == layer_assignment[pre], layer_assignment[post]] += probability_table[pre[0], post[0]]([pre[1], post[1]])
    return probability_table, layer_df


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
    excit = excit_pre.loc[:, excit_pre.columns.isin(
        ["PreSyn", "L2e", "L4e", "L5e"])]
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
        layers_only_df.loc[
            layers_only_df["PreSyn"] == syn[0], syn[1]] = np.sum(
                [a*b for a, b in zip(weights,
                                     [prob_e_e, prob_e_i, prob_i_e, prob_i_i])])
    # also, doing the second version of layers_only_df is hard because we add in the lefort data but its not accumulated -- its a float value. can go back into the dataset and get an n if its there. 
    return layers_only_df


def micro_errorplot(num_random_sims, assembly_size, num_assemblies):
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
    layers = ["L2", "L4", "L5"]
    layer_combos = list(itertools.product(layers, layers))
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
    average_random = np.median(list(chunked(prob_values, len(layer_combos))), axis=0)
    avg_random_df = pd.DataFrame(columns=layers)
    avg_random_df["PreSyn"] = layers
    for arv, (pre, post) in zip(average_random, layer_combos):
        avg_random_df.loc[avg_random_df["PreSyn"] == pre, post] = arv
    avg_random_slope, intercept, avg_random_r_value, p_value, std_err = linregress(
        real_for_df, average_random)
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
    se_slope = [np.sqrt((s[0] - 1)**2) for s in regression_fits]
    se_corr = [np.sqrt((r[1] - 1)**2) for r in regression_fits]
    closest_fits = np.argsort([s+c for s, c in zip(se_slope, se_corr)])
    rank_of_our_snmc = np.where(closest_fits == num_random_sims)
    return regression_fits, random_assignments, closest_fits, rank_of_our_snmc, [psp_agnostic, our_snmc_df, avg_random_df]


def compare_micro_connectivity_scatter(real_df, snmc_df, plotit):
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

def compare_micro_connectivity_heatmap(df_list, titles):
    maxval = np.max([np.nanmax(df.loc[:, ~df.columns.isin(["PreSyn"])].values) for df in df_list]) + .02
    fig, ax = pl.subplots(1, len(df_list))
    cmap = "BuPu"
    for i, (df, title) in enumerate(zip(df_list, titles)):
        if len(titles) > 1:
            sns.heatmap((df.loc[:, ~df.columns.isin(["PreSyn"])]).applymap(float),
                        yticklabels=df["PreSyn"], cmap=cmap, ax=ax[i], vmax=maxval)
            ax[i].set_title(title)
        else:
            sns.heatmap((df.loc[:, ~df.columns.isin(["PreSyn"])]).applymap(float),
                        yticklabels=df["PreSyn"], cmap=cmap, ax=ax, vmax=maxval)
            ax.set_title(title)
    pl.show()

    

# the next thing you have to do is establish the number of axons that exist in each layer
# there is one axon per cell per layer of connectivity. what you need is how many connections
# total (not probability) go from each unit to each unit. you have to write the above generator again so that the total number of axons and not the probability of connecting are encoded. these are not mappable to each other. you have to also add in the resampler, normalizer, and obssync.
# there is a clear difference in the way this has to be presented becuase there is a loop that returns to the source. this loop cannot exist if even one connection is broken. i still don't quite know how to condition on the existence of this loop.

# you just have to go with "connection is a branch" and branches are visible as axons. you can't say "an axon is one volume even if it connects to other neurons" because to get to the other neuron
# it has to create a volume. granted that volume might not be as large as the axon itself. have to think about this more, but as a first pass just do number of connections, because there are arguments to be made for both and number of connections is much easier.

""" ALLEN MACRO DATASET """ 


longrange = pd.read_csv("longrange.csv")

def assign_macro_components_to_brainregions():
    sample_score_components = ["MUX", "GATE", "TIK", "WTA", "Assemblies"]
    components = list(itertools.product(sample_score_components, ["P", "Q"])) + list(
        itertools.product(["Normalizer", "Resampler", "ObsStateSync"], [""]))
    components = [c for c in components if c != ("WTA", "P")]
    brain_regions = ["V1L2", "V1L4", "V1L5", "CP", "GPi", "LD"]
    random_brain_snmc_pairings = {c: np.random.choice(
        brain_regions) for c in components}
    random_brain_snmc_pairings[("DependentVariable", "")] = "S1"
    return random_brain_snmc_pairings, brain_regions + ["S1"]


def macro_axon_comparison(num_random_sims, assembly_size, num_assemblies):
    random_dfs = []
    random_assignments = []
    for s in range(num_random_sims):
        rand_br_assignment, brain_regions = assign_macro_components_to_brainregions()
        random_connections = generate_cross_component_axons(rand_br_assignment,
                                                            brain_regions,
                                                            num_assemblies,
                                                            assembly_size)
        random_dfs.append(random_connections)
        random_assignments.append(rand_br_assignment)

    sum_random_axons = reduce(lambda x, y: x + y.loc[:, y.columns!="PreSyn"].values, random_dfs, np.zeros(shape=(7,7)))
    sum_random_axons /= np.sum(sum_random_axons)
    average_random_df = pd.DataFrame(sum_random_axons, columns=random_dfs[0].columns[random_dfs[0].columns!="PreSyn"])
    average_random_df["PreSyn"] = random_dfs[0]["PreSyn"]

    snmc_br_assignment = {('MUX', 'P'): 'V1L5',
                          ('MUX', 'Q'): 'V1L5',
                          ('GATE', 'P'): 'V1L5',
                          ('GATE', 'Q'): 'V1L5',
                          ('TIK', 'P'): 'V1L5',
                          ('TIK', 'Q'): 'V1L5',
                          ('WTA', 'Q'): 'V1L2',
                          ('Assemblies', 'P'): 'V1L4',
                          ('Assemblies', 'Q'): 'V1L4',
                          ('Normalizer', ''): 'CP',
                          ('Resampler', ''): 'GPi',
                          ('ObsStateSync', ''): 'LD',
                          ('DependentVariable', ''): 'S1'}

    snmc_connections = generate_cross_component_axons(snmc_br_assignment,
                                                      brain_regions,
                                                      num_assemblies,
                                                      assembly_size)
    
    return snmc_assignments, random_assignments, random_dfs, average_random_df


def generate_cross_component_axons(brain_assignment, brain_regions,
                                   num_assemblies, assembly_size):
    size_obs = 3
    components = list(np.unique([b[0] for b in brain_assignment.keys()]))
    axon_table = {c: lambda pq: 0 for c in itertools.product(components,
                                                             components)}
    axon_table["WTA", "Assemblies"] = lambda pq: num_assemblies
    axon_table["WTA", "MUX"] = lambda pq: num_assemblies
    axon_table["WTA", "DownstreamVariable"] = lambda pq: num_assemblies
    axon_table["Assemblies", "WTA"] = lambda pq: 0 if pq[0] != pq[1] else (num_assemblies * assembly_size)
    axon_table["Assemblies", "MUX"] = lambda pq: (num_assemblies * assembly_size) if pq[0] == pq[1] else 0
    axon_table["Assemblies", "GATE"] = lambda pq: (num_assemblies * assembly_size) if pq[0] == pq[1] else 0
    axon_table["MUX", "TIK"] = lambda pq: num_assemblies if pq[0] == pq[1] == "Q" else 0
    axon_table["MUX", "GATE"] = lambda pq: num_assemblies if pq[0] == pq[1] == "P" else 0
    axon_table["TIK", "GATE"] = lambda pq: 1 if pq[0] == pq[1] else 0
    axon_table["GATE", "Normalizer"] = lambda pq: 1
    axon_table["Normalizer", "Resampler"] = lambda pq: 1
    axon_table["Resampler", "ObsStateSync"] = lambda pq: 1 + size_obs
    axon_table["WTA", "DependentVariable"] = lambda pq: num_assemblies
    axon_table["ObsStateSync", "DependentVariable"] = lambda pq: 1 + size_obs
    brain_region_df = pd.DataFrame(np.zeros(
        shape=(len(brain_regions), len(brain_regions))),
        columns=brain_regions)
    brain_region_df["PreSyn"] = brain_regions
    for pre, post in itertools.product(brain_assignment.keys(), brain_assignment.keys()):
        brain_region_df.loc[
            brain_region_df["PreSyn"] == brain_assignment[pre],
            brain_assignment[post]] += axon_table[pre[0], post[0]]([pre[1], post[1]])
    return brain_region_df
    






# next steps:
# 1) plots. how should these look? i like the idea of the directed graph, with arrow sizes or alpha corresponding to strength of connection, and a correlation fit to the overall values. that way you can quantitatively see that the fit is good, and qualitatively understand that the loop is predicted by SNMC. 

# how to properly integrate S1 into the diagram. we want to show that the V1 to S1 connection only exists if you put the injection into Layer 2. 



# Want V1 layer by layer to show that the component wise choices give rise to necessary macro connectivity.
# Want all loop members forward and backward.


# To make a statement about S1's connectivity to the loop, we have to follow a path from S1 to the loop and back. we can do this if it'll make the idea stronger. The main point of the S1 connection is to show that V1 layer choice matters. Probably the best idea, in the random assignment, to disallow anything other than V1 to connect to S1.

# This is a critical piece I think -- figure out how to best normalize the injection data and how to best normalize the S1 projection considering we are only thinking about it from an intercortical perspective.

# what we CAN say is that the parts of CP that V1 is connected to don't seem to pass back to S1.
# CP does not, neither does GPi. This is good! LD gives a touch to L6 S1, which makes sense.
# also note that the LD to CP is not real. They are almost definitely fibers of passage leaving the thalamus via the BG. We are not normalizing just saying "injections are approximately equal in volume, and may include off target circuitry". 

syns_of_interest = [("V1L2", "S1"), ("V1L4", "S1"), ("V1L5", "S1"),
                    ("V1L5", "CP"), ("V1L4", "CP"), ("V1L2", "CP"),
                    ("V1L5", "GPi"), ("V1L4", "GPi"), ("V1L2", "GPi"),
                    ("CP", "GPi"), ("CP", "V1"), ("CP", "LD"), ("CP", "S1"), 
                    ("GPi", "CP"), ("GPi", "LD"), ("GPi", "V1"), ("GPi", "S1"),
                    ("LD", "V1"), ("LD", "CP"), ("LD", "GPi"), ("LD", "S1")]
                    
