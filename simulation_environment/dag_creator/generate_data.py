import itertools
from functools import partial
from random import randint, random
import csv_functions as csv
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import semopy.model_generator as smg
import utility_functions
from numpy.random import normal, uniform
from scipy.interpolate import UnivariateSpline as sp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import randomized_causation_coefficient as rcc


def create_graph(n_lat, n_manif, p_manif, n_obs, n_cycles):
    m_model = smg.generate_measurement_part(n_lat, n_manif, p_manif)
    s_model, threadman = smg.generate_structural_part(m_model, n_obs, n_cycles)
    #TODO Right now I do not return the threadman, should I do this?
    return(m_model, s_model)

def generate_graphs(number, n_lat, n_manif, p_manif, n_obs, n_cycles):
    combinations = list(itertools.product(n_obs, n_lat, n_manif))
    for comb in combinations:
        for i in range(number // combinations):
            yield create_graph(comb[1], comb[2], p_manif, comb[0], n_cycles)

# Method to define exogenous variable, taken from Lopez-Paz
def cause(n,k,p1,p2):
  g = GaussianMixture(k, covariance_type = 'diag')
  g.means_   = p1*np.random.randn(k,1)
  g.covariances_  = np.power(abs(p2*np.random.randn(k,1)+1),2)
  g.weights_ = abs(np.random.rand(k))
  g.weights_ = g.weights_/sum(g.weights_)
  return scale(g.sample(n)[0])

# Method to define noise, taken from Lopez-Paz
# Noise is used in an additive way.
def noise(n,v):
    return v * np.random.rand(1) * np.random.randn(n, 1)


# Method to define a non-linear causal mechanism, taken from Lopez-Paz
def mechanism(x,d):
    g = np.linspace(min(x)-np.std(x),max(x)+np.std(x),d)
    # Fit a spline over g uniformly drawn points from the range of x, and d randomly drawn points from a normal
    # distribution. Uses x as input to calculate y.
    return(sp(g, np.random.randn(d), k=d-1)(x.flatten())[:,np.newaxis])

# Given a list of variables, calculates their values and the values of all
# their children, given that values of all their parents are known. Visits
# all children of every node, so calculating the value of every node is
# guaranteed.
def calculate_structure_values(vars, model, s_model_values, s_parents_dict,
                               n_samples, v, d):
    mms = MinMaxScaler()
    for var in vars:
        # Check if all parents already have a defined value and var is not
        # already defined.
        if (all(cause in s_model_values.keys() for cause in s_parents_dict[
            var])) and (var not in s_model_values):
            x = np.zeros([n_samples, 1])
            for parent in s_parents_dict[var]:
                # Add all parents * coefficients to create variables value.
                x = x + mechanism(s_model_values[parent], d)
            x = scale(x)
            # All noise variables have the same mean.
            s_model_values[var] = scale(x + noise(n_samples, v))
            # The vars we initially loop through are all exogenous variables,
            # so adding the children of their children will allow us to
            # traverse the whole tree.
            if var in model.keys():
                calculate_structure_values(model[var], model, s_model_values,
                                           s_parents_dict, n_samples, v, d)

# Function to generate non linear data in a pandas dataframe. Proces is split
#  in a structural and measurement part.
def generate_data_nonlinear(m_model, s_model, n_samples, n_mix, p1=10,p2=3,
                            v=0.001,d=5):

    exo_vars = set(s_model.keys()) - set([x for value in s_model.values() for x
                                        in value])
    exo_measure_vars = set(m_model.keys()) - set([x for value in
                                                  m_model.values() for x
                                                  in value])

    s_parents_dict = utility_functions.find_parents(s_model)
    m_parents_dict = utility_functions.find_parents(m_model)

    s_model_values = {}
    measure_vars = set([x for value in m_model.values() for x in value])
    measure_values = np.zeros([n_samples,len(measure_vars)])
    measure_vars = sorted(list(measure_vars))

    # If there is only a measurement model.
    if len(exo_vars) == 0:
        for exo in exo_measure_vars:
            s_model_values[exo] = cause(n_samples, n_mix, p1, p2)
            calculate_structure_values(m_model[exo], m_model, s_model_values,
                                       m_parents_dict, n_samples, v, d)

    # If there is a measurement and structural model.
    else:
        # Iterate through exogenous variables
        for exo in exo_vars:
            s_model_values[exo] = cause(n_samples, n_mix, p1, p2)
            calculate_structure_values(s_model[exo], s_model, s_model_values,
                                       s_parents_dict, n_samples, v, d)
        # Iterator object of all y's (measurement variables)
        for exo in exo_measure_vars:
            calculate_structure_values(m_model[exo], m_model, s_model_values,
                                   m_parents_dict, n_samples, v, d)

    for i, measure in enumerate(measure_vars):
        measure_values[:,i] = s_model_values[measure].flatten()
    return(pd.DataFrame(measure_values, columns=measure_vars))


# Output: (list of y values, labels of the combinations of variables, tetrads?)
# Generate a table with each row being a set of examples of four variables.
def generate_data(model,
                  n_samples, k, p1, p2, v, d):

    m_model, s_model = model

    # Can not give a m_model with more than 4y's to this function.
    labels = utility_functions.find_t_separations(m_model, s_model, ['y1','y2','y3','y4'])

    data = generate_data_nonlinear(m_model,
                                   s_model,
                                   n_samples,
                                   k,
                                   p1,
                                   p2,
                                   v,
                                   d)

    # TODO I disabled scaling.
    #list_of_ys = [single_val for val in sorted(set.union(*m_model.values())) for single_val in data[val].tolist()]
    return((data, labels))


# Goes through a set of 5 different parameters to generate extra training parameter.
# It is a good idea to first check one parameter combination a couple of times, if the scores for one set vary a lot, it is questionable how much
# it will mean to search for a specific parameter combination.
def find_best_training_parameters():
    # Number of sample per y.
    n_samples = 100

    # Amount of examples per variable in the test dataset. DO NOT CHANGE
    real_examples = 100

    n_sample_synthetic = 100 * 24

    #TODO I now only take the first example of each graph.
    models = utility_functions.get_graph_examples()

    #TODO csv file is hardcoded, should make this variable.

    # Generate synthetic data to extend training sets.
    real_values = pd.read_csv('simulated_data\spirtes_random0_samples100.csv')
    real_result = rcc.compare_distributions(real_values)
    scores = []
    count = 0

    # Find synthetic data parameters that come closest to real data.
    for k, d in itertools.product(*[[2, 3]] * 2):
        for p1, p2, v in itertools.product(*[[0.5, 2.5, 5]] * 3):
            graph_str = 'k{}_d{}_p1-{}_p2-{}_v{}_'.format(k, d, p1, p2, v)
            values_list = []
            for model in utility_functions.get_graph_examples():
                for m in range(int(n_sample_synthetic / len(models))):
                    values, target = generate_data(model, n_samples, k, p1, p2, v, d)
                    values_list.append(values)
            syn_result = rcc.compare_distributions(values_list, True)
            total_score = 0
            for i in range(real_result.shape[0]):
                best_score = 1000000000
                for j in range(syn_result.shape[0]):
                    result_dif = real_result[i, :] - syn_result[j, :]
                    score = np.dot(result_dif, result_dif)
                    if score < best_score: best_score = score
                total_score += best_score
            scores.append([graph_str, total_score])
            print(count)
            print(scores[count])
            count += 1

    print(scores)
    best_score = scores[0]
    for sc in scores:
        if sc[1] < best_score[1]: best_score = sc
    print("BEST")
    print(best_score)


def generate_extra_training_data(name, n_samples, n_sample_synthetic, k, d, p1, p2, v):
    models = utility_functions.get_graph_examples()
    for i in range(1):
        graph_str = 'k{}_d{}_p1-{}_p2-{}_v{}_{}'.format(k, d, p1, p2, v, name)
        csv.make_csv_predefmodel(n_samples, graph_str)
        for model in models:
            for m in range(int(n_sample_synthetic / len(models))):
                values, target = generate_data(model, n_samples, k, p1, p2, v, d)
                values = utility_functions.restructure_dataframe(values, model)
                csv.write_csv(values, target, graph_str)
