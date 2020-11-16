from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import utility_functions
from numpy.random import normal, uniform
import graph_examples
import transform_input_data
import csv_functions as csv
import itertools
from pathlib import Path

train_path = Path("generate_data.nosync/")
test_path = Path('simulated_data/')

def noise(n):
    #p = np.random.uniform(0.5,1)
    return(np.random.normal(0,1,n))

def s_model_mechanism(x, b):
    a = np.random.uniform(0.25,1)
    c = np.random.uniform(0.5,2)
    result = a * x + b * c * np.power(x,3)
    return(result)

def m_model_mechanism(x):
    coef = np.random.uniform(0.5,2)
    return(coef * x)

def m_model_nonlinear_mechanism(x, d):
    p = np.random.uniform(1,5)
    e = np.random.uniform(0.5, 2)
    f = np.random.uniform(0.5, 2)
    #p = np.random.randint(1, 4)
    return((1-d) * e * x + d * f * np.power(x,int(p)))

# Given a list of variables, calculates their values and the values of all
# their children, given that values of all their parents are known. Visits
# all children of every node, so calculating the value of every node is
# guaranteed.

def calculate_values(vars, model, s_model_values, s_parents_dict,
                               n_samples, s_model, b, d, linear):
    for var in vars:
        # Check if all parents already have a defined value and var is not
        # already defined.
        if (all(cause in s_model_values.keys() for cause in s_parents_dict[
            var])) and (var not in s_model_values):
            x = np.zeros([n_samples])
            for parent in s_parents_dict[var]:
                # Add all parents * coefficients to create variables value.
                if s_model:
                    x = x + s_model_mechanism(s_model_values[parent], b)
                else:
                    if linear:
                        x = x + m_model_mechanism(s_model_values[parent])
                    else:
                        x = x + m_model_nonlinear_mechanism(s_model_values[parent], d)
            x = scale(x)
            # All noise variables have the same mean.
            s_model_values[var] = scale(x + noise(n_samples))
            #s_model_values[var] = x + noise(n_samples)
            # The vars we initially loop through are all exogenous variables,
            # so adding the children of their children will allow us to
            # traverse the whole tree.
            if var in model.keys():
                calculate_values(model[var], model, s_model_values,
                                           s_parents_dict, n_samples, s_model, b, d, linear)

# Function to generate non linear data in a pandas dataframe. Proces is split
#  in a structural and measurement part.
def generate_data(m_model, s_model, n_samples, b, d, linear = True):

    exo_vars = set(s_model.keys()) - set([x for value in s_model.values() for x
                                        in value])
    exo_measure_vars = set(m_model.keys()) - set([x for value in
                                                  m_model.values() for x
                                                  in value])

    s_parents_dict = utility_functions.find_parents(s_model)
    m_parents_dict = utility_functions.find_parents(m_model)

    # The dictionary that will contain the value of all variables at the end.
    s_model_values = {}
    measure_vars = set([x for value in m_model.values() for x in value])
    measure_values = np.zeros([n_samples,len(measure_vars)])
    measure_vars = sorted(list(measure_vars))

    # Iterate through exogenous variables
    for exo in exo_vars:
        s_model_values[exo] = noise(n_samples)
        calculate_values(s_model[exo], s_model, s_model_values,
                                   s_parents_dict, n_samples, True, b, d, linear)
    # Iterator object of all y's (measurement variables)
    for exo in exo_measure_vars:
        calculate_values(m_model[exo], m_model, s_model_values,
                               m_parents_dict, n_samples, False, b, d, linear)

    for i, measure in enumerate(measure_vars):
        measure_values[:,i] = s_model_values[measure].flatten()
    return(pd.DataFrame(measure_values, columns=measure_vars))

# generate data as a normal table.
def generate_data_linear(n_samples, b, d, test_size, model = graph_examples.exampleSpirtes()):
    m_model, s_model = model
    n = 0
    while n < test_size:
        data = generate_data(m_model, s_model, n_samples, b, d, True)
        corr = data.corr()
        no_obvious_examples = True
        for var in corr.columns:
            if not ((corr[var].drop(var) >= 0.09).all() and (corr[var].drop(var) <= 0.90).all()):
                no_obvious_examples = False
        if no_obvious_examples:
            data.to_csv(test_path / 'spirtes_random_b{}_d{}_samples{}_n{}.csv'.format(b, d, n_samples, n),
                    index=False)
            n += 1

def generate_data_nonlinear(n_samples, b, d, test_size, model = graph_examples.exampleSpirtes()):
    m_model, s_model = model
    n = 0
    while n < test_size:
        data = generate_data(m_model, s_model, n_samples, b, d, False)
        corr = data.corr()
        no_obvious_examples = True
        for var in corr.columns:
            if not ((corr[var].drop(var) >= 0.09).all() and (corr[var].drop(var) <= 0.90).all()):
                no_obvious_examples = False
        if no_obvious_examples:
            data.to_csv(test_path / 'spirtes_nonlin_random_b{}_d{}_samples{}_n{}.csv'.format(b, d, n_samples,
                                                                                             n), index=False)
            n += 1


def generate_data_both(n_samples, b, d):
    generate_data_nonlinear(n_samples, b, d, graph_examples.exampleSpirtes())
    generate_data_linear(n_samples, b, d, graph_examples.exampleSpirtes())

# Generate a csv of which each row represents an x amount of observations of four variables
def generate_data_multiple_distributions(n_samples, n_dist, b, d, linearity, purity = True):
    graph_str = 'multiple_distributions_Spirtes_'
    if purity:
        models_list = utility_functions.get_graph_examples()
    else:
        models_list = utility_functions.get_graph_examples_impure()
    csv.make_csv_predefmodel(n_samples, graph_str)
    for models in models_list:
        for model in models:
            if len(models) > n_dist: r = range(1)
            else: r = range(int(n_dist / len(models)))
            for m in r:
                m_model, s_model = model
                values = sorted(set.union(*m_model.values()))
                combinations = itertools.combinations(values,4)
                values = generate_data(m_model, s_model, n_samples, b, d, linearity)
                for comb in combinations:
                    target = utility_functions.find_t_separations(m_model, s_model, comb)
                    value = utility_functions.restructure_dataframe_specific_vars(values, comb)
                    csv.write_csv(value, target, graph_str)

def generate_data_multiple_distributions_complex_graph(n_samples, n_sets, b, d, nonlinearity, model=graph_examples.exampleSpirtes_simpel()):
    graph_str = 'multiple_distributions_Spirtes_complex_graph_'
    csv.make_csv_predefmodel(n_samples, graph_str)
    for m in model:
        m_model, s_model = m
        for n in range(n_sets):
            values = generate_data(m_model, s_model, n_samples, b, d, nonlinearity)
            for comb in list(itertools.combinations(set([str(val) for vals in m_model.values() for val in vals]), 4)):
                target = utility_functions.find_t_separations(m_model, s_model, comb)
                value = utility_functions.restructure_dataframe_specific_vars(values, comb)
                csv.write_csv(value, target, graph_str)




