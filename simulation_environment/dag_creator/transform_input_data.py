import csv
import os
import pandas as pd
import utility_functions as util_f
from sklearn.preprocessing import scale
import itertools
import graph_examples

def ready_csv(dir_str, name, values_len, var_len):
    with open(dir_str + name + '_values.csv', 'w+', newline='') as value_file, \
            open(dir_str + name + '_targets.csv', 'w+', newline='') as target_file:
                value_writer = csv.writer(value_file, delimiter=',')
                target_writer = csv.writer(target_file, delimiter=',')
                # Add four for the variables.
                value_writer.writerow(list(range((values_len * 4) + var_len)))
                # six for 3 times 2 two variable combinations and an additional three for the label of every tetrad.
                target_writer.writerow(list(range(9)))

# values_pd is a pd dataframe.
def append_to_csv(dir_str, name, values_pd, var, t_sep):

    with open(dir_str + name + '_values.csv', 'a',newline='') as value_file, \
            open(dir_str + name + '_targets.csv', 'a', newline='') as target_file:
                value_writer = csv.writer(value_file, delimiter=',')
                target_writer = csv.writer(target_file, delimiter=',')
                values = list(scale(values_pd[var[0]])) + list(scale(values_pd[var[1]])) \
                         + list(scale(values_pd[var[2]])) + list(scale(values_pd[var[3]])) \
                         + list(var)

                value_writer.writerow(values)
                target_writer.writerow(t_sep)

def ready_target_csv(dir_str, name):
    with open(dir_str + name + '_targets.csv', 'w+', newline='') as target_file:
                target_writer = csv.writer(target_file, delimiter=',')
                # Add four for the variables.
                # six for 3 times 2 two variable combinations and an additional three for the label of every tetrad.
                target_writer.writerow(list(range(9)))

def append_to_target_csv(dir_str, name, t_sep):

    with open(dir_str + name + '_targets.csv', 'a', newline='') as target_file:
                target_writer = csv.writer(target_file, delimiter=',')
                target_writer.writerow(t_sep)

def political_democracy():
    s_model = {}
    m_model = {'ind60': {'dem60', 'dem65', 'x1', 'x2', 'x3'},
               'dem60': {'y1','y2','y3','y4','dem65'},
               'dem65': {'y5','y6','y7','y8'}}
    variables = set.union(*m_model.values()) - m_model.keys()
    var_comb = list(itertools.combinations(variables, 4))
    dir_str = os.getcwd().replace('\\', '/') + '/real_data/'
    name = 'poldem'
    poldem = pd.read_csv(dir_str + 'politicaldemocracy.csv')
    # Length of first list of variables dictates the amount in the CSV file.
    ready_csv(dir_str, name, poldem.shape[0], len(var_comb[0]))

    for variable in var_comb:
        t_separations = util_f.find_t_separations(m_model, s_model, variable)
        append_to_csv(dir_str, name, poldem, variable, t_separations)

def spirtes_data(m_model, s_model):
    variables = set.union(*m_model.values())
    var_comb = list(itertools.combinations(variables, 4))
    name = 'spirtes_tetrad_constraints'
    # Length of first list of variables dictates the amount in the CSV file.
    dir_str = os.getcwd().replace('\\', '/') + '/simulated_data/'
    ready_target_csv(dir_str, name)

    for variable in var_comb:
        t_separations = util_f.find_t_separations(m_model, s_model, variable)
        append_to_target_csv(dir_str, name, t_separations)