import csv
import os
import pathlib

def make_csv_predefmodel(n_samples, csv_name=''):
    dir_str = os.getcwd().replace('\\', '/') + '/generated_data.nosync/'
    with open(dir_str + csv_name + 'gen_values.csv', 'w+',
              newline='') as value_file, \
        open(dir_str + csv_name + 'gen_targets.csv', 'w+', newline='') as target_file:
        value_writer = csv.writer(value_file, delimiter=',')
        target_writer = csv.writer(target_file, delimiter=',')
        value_writer.writerow(list(range(n_samples * 4)))
        target_writer.writerow(list(range(9)))

def write_csv(values, target, csv_name=''):
    # Can also use pd.to_csv()
    # with open('generated_data.nosync/' + csv_name + 'gen_values.csv', 'w',
    #           newline='') as value_file, open('generated_data.nosync/' + csv_name +
    #     'gen_targets.csv', 'w', newline='') as target_file, open('generated_data.nosync/' + csv_name +
    #     'gen_tetrads.csv', 'w', newline='') as tetrad_file:        \
    dir_str = os.getcwd().replace('\\', '/') + '/generated_data.nosync/'
    with open(dir_str + csv_name + 'gen_values.csv', 'a',
              newline='') as value_file, \
        open(dir_str + csv_name + 'gen_targets.csv', 'a', newline='') as target_file:
        value_writer = csv.writer(value_file, delimiter=',')
        target_writer = csv.writer(target_file, delimiter=',')
        value_writer.writerow(values)
        target_writer.writerow(target)

def exp_make_csv_predefmodel(header, csv_name=''):
    dir_str = os.getcwd().replace('\\', '/') + '/experiment_results_local/'
    with open(dir_str + csv_name + '_results.csv', 'w+',
              newline='') as result_file:
        value_writer = csv.writer(result_file, delimiter=',')
        value_writer.writerow(header)

def exp_write_csv(values, csv_name=''):
    # Can also use pd.to_csv()
    # with open('generated_data.nosync/' + csv_name + 'gen_values.csv', 'w',
    #           newline='') as value_file, open('generated_data.nosync/' + csv_name +
    #     'gen_targets.csv', 'w', newline='') as target_file, open('generated_data.nosync/' + csv_name +
    #     'gen_tetrads.csv', 'w', newline='') as tetrad_file:        \
    dir_str = os.getcwd().replace('\\', '/') + '/experiment_results_local/'
    with open(dir_str + csv_name + '_results.csv', 'a',
              newline='') as result_file:
        value_writer = csv.writer(result_file, delimiter=',')
        value_writer.writerow(values)
