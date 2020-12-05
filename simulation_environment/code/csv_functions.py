import csv
import os
import pickle
import base64
import h5py


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


def kme_make_h5(kme_list, path, name=''):
    f = h5py.File(path / (name + '_kme_data.hdf5'), "w")
    for n, kme in enumerate(kme_list):
        x, y, weights = kme
        w, wz, wp, wp1, wp2 = weights
        n_group = f.create_group(str(n))
        n_group.create_dataset('x',data=x)
        n_group.create_dataset('y',data=y)
        n_group.create_dataset('w',data=w)
        n_group.create_dataset('wz',data=wz)
        n_group.create_dataset('wp',data=wp)
        n_group.create_dataset('wp1',data=wp1)
        n_group.create_dataset('wp2',data=wp2)

def kme_read_h5(n, path, name):
    f = h5py.File(path / (name + '_kme_data.hdf5'), "r")
    x = f['{}/x'.format(n)]
    y = f['{}/y'.format(n)]
    weights = (f['{}/w'.format(n)], f['{}/wz'.format(n)], f['{}/wp'.format(n)], f['{}/wp1'.format(n)],
               f['{}/wp2'.format(n)])
    return(x, y, weights)


