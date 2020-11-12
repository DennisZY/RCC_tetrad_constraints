import pickle
from graphviz import Digraph
import numpy as np
from t_separation import is_t_separated
import graph_examples as ge
import pandas as pd
import itertools

# Return a dictionary with (variable,data) key value pairs.
def get_values(values_df, i, variables):
    # Amount of each y in a row.
    vl = round(values_df.shape[1] / 4)
    var1 = np.array(values_df.iloc[i][0:vl])
    var2 = np.array(values_df.iloc[i][vl:vl * 2])
    var3 = np.array(values_df.iloc[i][vl * 2:vl * 3])
    var4 = np.array(values_df.iloc[i][vl * 3:vl * 4])
    values = pd.DataFrame(np.array([var1,var2,var3,var4]).T, columns=variables)
    return(values)

def restructure_dataframe(df, model):
    m_model, s_model = model
    return([single_val for val in sorted(set.union(*m_model.values())) for single_val in df[val].tolist()])

def restructure_dataframe_specific_vars(df, vars):
    return([single_val for val in vars for single_val in df[val].tolist()])

# Source: https://stackoverflow.com/questions/5360220/how-to-split-a-list-into-pairs-in-all-possible-ways
# All three unique ways to pick 2 groups of 2 out of 4.
def all_pairs(lst):
    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        # Chances are that this odd length list feature does not work.
        # Handle odd length list
        for i in range(len(lst)):
            for result in all_pairs(lst[:i] + lst[i+1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1,len(lst)):
            pair = (a,lst[i])
            for rest in all_pairs(lst[1:i]+lst[i+1:]):
                yield [pair] + rest

def tetrad_pairs(y1, y2, y3, y4):
    return([
        [[y1, y2], [y3, y4]],
        [[y1, y3], [y2, y4]],
        [[y1, y4], [y2, y3]]])


# TODO function takes m_model for its values, but the function only works on
# y variables at a time and not all y's at the same time.
def find_t_separations(m_model, s_model, comb=None):
    nodes = set.union(*m_model.values(), m_model.keys(), *s_model.values(),
                      s_model.keys())
    index_dict = {}
    for i, node in enumerate(nodes):
        index_dict[node] = i
    d = len(nodes)
    mL = np.zeros((d, d), dtype=bool)
    m0 = np.eye(d, dtype=bool)
    for k in m_model.keys():
        for v in m_model[k]:
            mL[index_dict[k], index_dict[v]] = True
    for k in s_model.keys():
        for v in s_model[k]:
            mL[index_dict[k], index_dict[v]] = True
    graph = (mL, m0)

    # Option to take the t-separation constraints from 1 combination of the model, or all variable combinations in the model.
    if comb == None:
        t_sep = []
        for comb in itertools.combinations(set.union(*m_model.values()), 4):
            t_separations = build_t_sep_string(comb, index_dict, graph, d)
            t_sep.append(t_separations)
    else:
        t_sep = build_t_sep_string(comb, index_dict, graph, d)
    return(t_sep)

def build_t_sep_string(comb, index_dict, graph, d):
    t_separations = []
    for set_of_pairs in tetrad_pairs(comb[0], comb[1], comb[2], comb[3]):
        A = np.zeros(d, dtype=bool)
        B = np.zeros(d, dtype=bool)
        A[index_dict[set_of_pairs[0][0]]] = A[index_dict[set_of_pairs[0][1]]] = True
        B[index_dict[set_of_pairs[1][0]]] = B[index_dict[set_of_pairs[1][1]]] = True
        set_of_pairs = sorted((sorted(set_of_pairs[0]), sorted(set_of_pairs[1])))
        t_separations.append(set_of_pairs[0][0] + " " + set_of_pairs[0][1])
        t_separations.append(set_of_pairs[1][0] + " " + set_of_pairs[1][1])
        t_separations.append(is_t_separated(graph, A, B))
    return(t_separations)

def calculate_tetrad_constraints(covar_matrix):
    tetrad_list = []
    tetrad_list.append(covar_matrix.loc['y1', 'y2'] *
                       covar_matrix.loc['y3', 'y4'] -
                       covar_matrix.loc['y1', 'y3'] *
                       covar_matrix.loc['y2', 'y4'])
    tetrad_list.append(covar_matrix.loc['y1', 'y2'] *
                       covar_matrix.loc['y3', 'y4'] -
                       covar_matrix.loc['y1', 'y4'] *
                       covar_matrix.loc['y2', 'y3'])
    tetrad_list.append(covar_matrix.loc['y1', 'y3'] *
                       covar_matrix.loc['y2', 'y4'] -
                       covar_matrix.loc['y1', 'y4'] *
                       covar_matrix.loc['y2', 'y3'])
    return(tetrad_list)

# Helper function related to graphs.

def get_graph_count():
    return(ge.graph_count())

def get_graph_examples(examples=list(range(ge.graph_count()))):
    example_list = []
    example_list.append(ge.example0())
    example_list.append(ge.example1())
    return(example_list)

    # example_list = []
    # for i in examples:
    #     graph_list = eval('ge.example' + str(i))()
    #     for g in graph_list:
    #         example_list.append(g)
    # return(example_list)

def get_graph_examples_impure(examples=list(range(ge.graph_count()))):
    example_list = []
    example_list.append(ge.example0())
    example_list.append(ge.example1())
    example_list.append(ge.example_cross_construct_impure())
    example_list.append(ge.example_intra_construct_impure())
    example_list.append(ge.example_latent_measure_impure())
    return(example_list)


def visualize_graph(m_model, s_model):
    nodes = set.union(*m_model.values(), m_model.keys(), *s_model.values(),
                      s_model.keys())
    G = Digraph()
    for node in nodes:
        G.node(node)
    for k in m_model.keys():
        for v in m_model[k]:
            G.edge(k, v)
    for k in s_model.keys():
        for v in s_model[k]:
            G.edge(k, v)
    return(G)

def find_parents(model):
    parents_dict = {}
    for k in model.keys():
        for v in model[k]:
            if v in parents_dict.keys():
                parents_dict[v].append(k)
            else:
                parents_dict[v] = [k]
    return(parents_dict)

def find_descendants(origin, model):
    descendants = []
    if origin in model.keys():
        for v in model[origin]:
            find_descendants_rec(v, model, descendants)
    return (descendants)

def find_descendants_rec(origin, model, descendants):
    descendants.append(origin)
    if origin in model.keys():
        for v in model[origin]:
            find_descendants_rec(v, model, descendants)
