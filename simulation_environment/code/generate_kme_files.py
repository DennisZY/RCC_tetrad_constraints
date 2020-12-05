import randomized_causation_coefficient as rcc
import generate_data
import pandas as pd
import itertools
from pathlib import Path
import time
import graph_examples
import csv_functions as csv
import transform_input_data

test_path = Path('simulated_data/')
file_path_non_linear = Path('kme_data/non_linear')
file_path_linear = Path('kme_data/linear')

def generate_KME(list_b, list_d, list_K, list_KME, list_nsamples, test_size, model):
    transform_input_data.spirtes_data(*model)
    t0 = time.time()
    count = 0

    test_target = pd.read_csv(test_path /
                              'spirtes_tetrad_constraints_targets.csv')

    for prod in itertools.product(list_nsamples,list_b,list_d):
        nsamp, b, d = prod
        # generate 10 files with test distributions.
        generate_data.generate_data_nonlinear(nsamp, b, d, test_size, model)
        for product in list(itertools.product(list_KME, list_K)):
            count += 1
            print('Count: {}'.format(count))
            KME, K = product
            kme_file_name = 'b{}_d{}_samples{}_K{}_KME{}'.format(b, d, nsamp, K, KME)

            print('{} & {}'.format(product, prod))

            kme_list = []
            for n in range(test_size):
                w = rcc.create_weights(K)

                test_val = pd.read_csv(test_path / 'spirtes_nonlin_random_b{}_d{}_samples{}_n{}.csv'.format(b,d,nsamp,n))

                x, y = rcc.kernel_mean_embedding(test_val, test_target, w, False, KME)

                kme_list.append((x,y,w))

            csv.kme_make_h5(kme_list, file_path_non_linear, kme_file_name)

            print("Finished at time:")
            print(time.time() - t0)
            t0 = time.time()
            print()