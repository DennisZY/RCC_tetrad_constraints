import randomized_causation_coefficient as rcc
import generate_data_Spirtes
import numpy as np
import time
import itertools
import csv_functions as csv
import graph_examples
from statistic_test import test_data, test_data_single
import pandas as pd
import statistics
import transform_input_data
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix

train_path = Path("generated_data.nosync/")
test_path = Path('simulated_data/')

def main():
    spirtes_wishart_poldem()

def poldem():
    # gd.generate_extra_training_data('spirtes', 100, 2400, 2, 3, 2.5, 0.5, 5)
    k = 1
    d = 5
    p1 = 2.5
    p2 = 5
    v = 0.5
    gd.generate_extra_training_data('poldem', 75, 24 * 400, k, d, p1, p2, v)

    train_val_path = 'generated_data.nosync\k{}_d{}_p1-{}_p2-{}_v{}_poldemgen_values.csv'.format(k, d, p1, p2, v)
    train_target_path = 'generated_data.nosync\k{}_d{}_p1-{}_p2-{}_v{}_poldemgen_targets.csv'.format(k, d, p1, p2, v)
    test_val_path = 'real_data\politicaldemocracy.csv'
    test_target_path = 'real_data\poldem_targets.csv'

    rcc.test_spirtes(train_val_path, train_target_path, test_val_path, test_target_path)

def spirtes():
    k = 2
    d = 5
    p1 = 2.5
    p2 = 5
    v = 0.5
    #gd.generate_extra_training_data('spirtes', 100, 24 * 350, k, d, p1, p2, v)
    #generate_data_Spirtes.generate_data_basic()

    #generate_data_Spirtes.generate_data_multiple_distributions(100, 100 * 24, random)

    #print(rcc.median_heuristic(pd.read_csv('simulated_data\spirtes_random0_samples1000.csv')))

    #train_val_path = 'generated_data.nosync\k{}_d{}_p1-{}_p2-{}_v{}_spirtesgen_values.csv'.format(k, d, p1, p2, v)
    #train_target_path = 'generated_data.nosync\k{}_d{}_p1-{}_p2-{}_v{}_spirtesgen_targets.csv'.format(k, d, p1, p2, v)
    train_val_path = 'generated_data.nosync\multiple_distributions_Spirtesgen_values.csv'
    train_target_path = 'generated_data.nosync\multiple_distributions_Spirtesgen_targets.csv'

    test_target_path = 'simulated_data\spirtes_tetrad_constraints_targets.csv'
    t0 = time.time()

    # Things to vary:
    # embedding
    # Length of KME vector
    # Amount of trees in RFC
    # Amount of non-linearity.
    random = 0.05
    E = [500, 1000, 1500]
    K = [200, 400, 800]
    KME = ['minimal', '4', 'marginal']
    csv.exp_make_csv_predefmodel(['KME','E', 'K','n_samples','n_distributions','best_score', 'mean_score', 'var_score'], str(random))

    for product in list(itertools.product(KME, E, K)):
        KME, E, K = product
        print(product)

        test_val_path = 'simulated_data\spirtes_random{}_samples5000.csv'.format(random)
        rcc.test_spirtes(train_val_path, train_target_path, test_val_path, test_target_path, str(random), E, K, KME)
        print("Finished at time:")
        print(time.time() - t0)
        print()

def spirtes_nonlin():
    # Things to vary:
    # embedding
    # Length of KME vector
    # Amount of trees in RFC
    # Amount of non-linearity.
    model = graph_examples.exampleSpirtes_simpel()
    transform_input_data.spirtes_data(*model)

    linear_train = [False]
    list_b = [0.05]
    list_d = [0.05]
    path = 'experiment_larget_sample_trees_weights'
    list_E = [500]
    list_K = [400]
    list_KME = ['4'] #['minimal', '4', 'marginal']
    list_nsamples = [2000]
    list_ndistributions = [4000]

    #generate_data_Spirtes.generate_data_multiple_distributions_complex_graph(500, 1000, random, True, model=graph_examples.exampleSpirtes_minimal())

    t0 = time.time()

    csv.exp_make_csv_predefmodel(['train_lin','b','d','KME','E', 'K','n_samples','n_distributions','score','trueneg','falseneg','truepos','falsepos'], path)
    for prod in itertools.product(list_nsamples, list_ndistributions,list_b,list_d):
        nsamp, ndist, b, d = prod
        generate_data_Spirtes.generate_data_multiple_distributions(nsamp, ndist, b, d, linear_train)
        # generate 10 files with test distributions.
        generate_data_Spirtes.generate_data_nonlinear(nsamp, b, d, model)
        train_val = pd.read_csv(
            train_path / 'multiple_distributions_Spirtes_gen_values.csv')
        train_target = pd.read_csv(
            train_path / 'multiple_distributions_Spirtes_gen_targets.csv')
        test_target = pd.read_csv(test_path /
                                  'spirtes_tetrad_constraints_targets.csv')
        for product in list(itertools.product(list_KME, list_E, list_K)):
            KME, E, K = product
            w = rcc.create_weights(K)

            print('{} & {}'.format(product, prod))

            x1, y1 = rcc.kernel_mean_embedding(train_val, train_target, w, True, KME)
            reg = RFC(n_estimators=E, random_state=0, n_jobs=16).fit(x1, y1)
            print("RFC fitted")

            for n in range(10):
                test_val = pd.read_csv(test_path / 'spirtes_nonlin_random_b{}_d{}_samples{}_n{}.csv'.format(b,d,nsamp,n))

                x2, y2 = rcc.kernel_mean_embedding(test_val, test_target, w, False, KME)
                prediction = reg.predict(x2)
                cm = confusion_matrix(y2, prediction)
                trueneg = cm[0,0]
                falseneg = cm[1,0]
                truepos = cm[1,1]
                falsepos = cm[0,1]
                score = reg.score(x2, y2)
                print('Score: {}'.format(score))
                csv.exp_write_csv([linear_train,b,d,KME, E, K, nsamp, ndist, score,trueneg,falseneg,truepos,falsepos],
                          path)
            print("Finished at time:")
            print(time.time() - t0)
            print()

def spirtes_wishart():
    model = graph_examples.exampleSpirtes_simpel()
    list_b = [0.01,0.05]
    list_d = [0.01,0.05]
    list_n_samples = [200,500,1000,2000,10000]
    test_target_path = test_path / 'spirtes_tetrad_constraints_targets.csv'
    targets = pd.read_csv(test_target_path)
    filename = 'wishart_experiment_samplesize_vanilla'
    csv.exp_make_csv_predefmodel(['linear','b','d','nsamples','accuracy','trueneg','falseneg','truepos','falsepos'],filename)
    for product in itertools.product(list_n_samples, list_b, list_d):
        n_samples, b, d = product
        #generate_data_Spirtes.generate_data_nonlinear(n_samples, b, d, model)
        for n in range(100):
            values = pd.read_csv(test_path / 'spirtes_nonlin_random_b{}_d{}_samples{}_n{}.csv'.format(b, d, n_samples, n))

            acc, tetrad_list, label_list = test_data_single(values, targets)
            cm = confusion_matrix(label_list, tetrad_list)
            trueneg = cm[0, 0]
            falseneg = cm[1, 0]
            truepos = cm[1, 1]
            falsepos = cm[0, 1]

            csv.exp_write_csv([False, b, d, n_samples, acc, trueneg, falseneg, truepos, falsepos], filename)

    #This should test the Wishart test in the linear case.
    list_b = [0,0.1,0.05]
    list_d = [0]
    for product in itertools.product(list_n_samples, list_b, list_d):
        n_samples, b, d = product
        generate_data_Spirtes.generate_data_linear(n_samples, b, d, model)
        for n in range(100):
            values = pd.read_csv(test_path / 'spirtes_random_b{}_d{}_samples{}_n{}.csv'.format(b, d, n_samples, n))

            acc, tetrad_list, label_list = test_data_single(values, targets)
            cm = confusion_matrix(label_list, tetrad_list)
            trueneg = cm[0, 0]
            falseneg = cm[1, 0]
            truepos = cm[1, 1]
            falsepos = cm[0, 1]

            csv.exp_write_csv([True, b,d, n_samples, acc, trueneg,falseneg,truepos,falsepos],filename)

def spirtes_wishart_poldem():
    targets = pd.read_csv(Path('real_data') / 'poldem_targets.csv')
    values = pd.read_csv(Path('real_data') / 'politicaldemocracy.csv')

    #filename = 'wishart_experiment_poldem'

    acc, tetrad_list, label_list = test_data_single(values, targets)
    cm = confusion_matrix(label_list, tetrad_list)
    trueneg = cm[0, 0]
    falseneg = cm[1, 0]
    truepos = cm[1, 1]
    falsepos = cm[0, 1]
    print(cm)

    true_pos = truepos / (truepos + falseneg)
    false_pos = falsepos / (falsepos + trueneg)

    print(true_pos)
    print(false_pos)


if __name__ == "__main__":
    main()