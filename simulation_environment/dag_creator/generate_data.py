import semopy.model_generator as smg
import utility_functions
import numpy as np
import pandas as pd
from functools import partial
from numpy.random import normal, uniform
import itertools
import csv
from random import randint, random
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.mixture       import GaussianMixture
from collections import Counter
from scipy.interpolate     import UnivariateSpline as sp


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
  #return scale(g.sample(n)[0])
  return(np.random.normal(0, 1, (n,1)))
  #return(g.sample(n)[0])

# Method to define noise, taken from Lopez-Paz
# Noise is used in an additive way.
def noise(n,v):
    #return(np.zeros([n,1]))
    #return(v*np.random.randn(n,1))
    return(v * np.random.normal(0, 1, (n, 1)))
    #return(v * np.random.normal(np.random.normal(0,50), np.random.randint(5,10), (n, 1)))

# Method to define a non-linear causal mechanism, taken from Lopez-Paz
def mechanism(x,d):
  g = np.linspace(min(x)-np.std(x),max(x)+np.std(x),d);
  return(sp(g,np.random.randn(d))(x.flatten())[:,np.newaxis])

def linear_mechanism(x,d):
    #lin_mech = x * (randint(-15,15) + random())
    coef = randint(-5,5) + random()
    lin_mech = x * coef
    return(lin_mech)

# Given a list of variables, calculates their values and the values of all
# their children, given that values of all their parents are known. Visits
# all childer of every node, so calculating the value of every node is
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
                x = x + linear_mechanism(s_model_values[parent], d)
            # All noise variables have the same mean.
            #x = mms.fit_transform(x)
            s_model_values[var] = x + noise(n_samples, v)
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

def generate_data(model,
                  n_samples,
                  mpart_generator,
                  spart_generator,
                  data_generator,
                  data_error_generator,
                  threadman=None,
                  scale_g = 1,
                  data_errors_post = True):

    m_model, s_model = model

    # Can not give a m_model with more than 4y's to this function.
    labels = utility_functions.find_t_separations(m_model, s_model)

    data = generate_data_nonlinear(m_model,
                                   s_model,
                                   n_samples,
                                   5)

    tetrads = utility_functions.calculate_tetrad_constraints(data.cov())

    #data = pd.DataFrame(scale(data), columns=data.columns)

    # m_param, s_param = smg.generate_parameters(m_model,
    #                                            s_model,
    #                                            mpart_generator,
    #                                            spart_generator,
    #                                            mpart_fix_value = None)

    # data = smg.generate_data(m_model,
    #                          s_model,
    #                          m_param,
    #                          s_param,
    #                          n_samples,
    #                          threadman,
    #                          generator = partial(normal, normal(0,1), 1),
    #                          error_generator = data_error_generator,
    #                          errors_post = data_errors_post)

    # return a list containing all values of [y1, y2, y3, y4] in this
    # specific order.
    # TODO I disabled scaling.
    mms = MinMaxScaler()
    #data = pd.DataFrame(mms.fit_transform(data), columns = data.columns)
    list_of_ys = [single_val for val in sorted(set.union(*m_model.values()))
                  for single_val in data[val].tolist()]

    return((list_of_ys, labels, tetrads))

def make_csv_predefmodel(n_sample_sets, n_samples,
                         mpart_generator, spart_generator, data_generator,
                         data_error_generator, models, csv_name=''):
    # Can also use pd.to_csv()
    with open('generated_data.nosync/' + csv_name + 'gen_values.csv', 'w',
              newline='') as value_file, open('generated_data.nosync/' + csv_name +
        'gen_targets.csv', 'w', newline='') as target_file, open('generated_data.nosync/' + csv_name +
        'gen_tetrads.csv', 'w', newline='') as tetrad_file:        \

        value_writer = csv.writer(value_file, delimiter=',')
        target_writer = csv.writer(target_file, delimiter=',')
        tetrad_writer = csv.writer(tetrad_file, delimiter=',')

        value_writer.writerow(list(range(n_samples * 4)))
        target_writer.writerow(list(range(9)))
        tetrad_writer.writerow(list(range(3)))
        for model in models:
            for i in range(int(n_sample_sets / len(models))):
                values, target, tetrads = generate_data(model,
                                               n_samples, mpart_generator,
                                               spart_generator, data_generator,
                                               data_error_generator)
                value_writer.writerow(values)
                target_writer.writerow(target)
                tetrad_writer.writerow(tetrads)

# Total number of variables in the structural part
n_obs = [1]

# Number of latent variables in the structural part
n_lat = [1]

# Minimal number of cycles in the structural part
n_cycles = 0

# Lower number of possible numbers of manifest variables for a latent variable
l_manif = 4
# Upper number of possible numbers of manifest variables for a latent variable
u_manif = 4
# Manifest variables
n_manif = [(l_manif, u_manif)]
# fraction of manifest variables to merge together
p_manif = 0.00
# Number of sample per y.
n_samples = 500

mpart_generator = lambda:  uniform(1,10)

spart_generator = lambda:  uniform(0.1,5)

data_generator = partial(normal, normal(0,1), 1)

data_error_generator = partial(normal, 0, np.sqrt(0.1))

graph_example_count = utility_functions.get_graph_count()

# number of sample datasets, should be a multiple of 72 (the amount of
# different graphs)
n_sample_sets = 20 * 72

n_sample_sets_test = 4 * 48

#TODO I now only take the first example of each graph.
models = utility_functions.get_graph_examples()

# model0 = utility_functions.get_graph_examples([0])[0]
# model1 = utility_functions.get_graph_examples([1])[0]
# model2 = utility_functions.get_graph_examples([2])[0]
# models = [model0,model1,model2]

# Generate training data.
make_csv_predefmodel(n_sample_sets, n_samples, mpart_generator,
         spart_generator, data_generator, data_error_generator,
         models)

# TODO disable test data generation.
#Generate test data.
for i in range(graph_example_count):
    models = utility_functions.get_graph_examples([i])
    make_csv_predefmodel(24, n_samples, mpart_generator,
                         spart_generator, data_generator, data_error_generator,
                         models, 'graph{}_'.format(i))
