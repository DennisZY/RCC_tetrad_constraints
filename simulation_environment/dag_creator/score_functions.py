import randomized_causation_coefficient as rcc
import generate_data_Spirtes
import time
import itertools
import csv_functions as csv
import numpy as np
from statistic_test import test_data, test_data_single
import pandas as pd
import transform_input_data
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix

train_path = Path("generated_data.nosync/")
test_path = Path('simulated_data/')

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


def spirtes_nonlin(linear_train, list_b, list_d, list_E, list_K, list_KME, list_nsamples, test_size,
                   list_ndistributions,path, model, impure_train=False):
    # Things to vary:
    # embedding
    # Length of KME vector
    # Amount of trees in RFC
    # Amount of non-linearity.

    t0 = time.time()
    count = 0
    csv.exp_make_csv_predefmodel(['train_lin','b','d','KME','E', 'K','n_samples','n_distributions','score','trueneg','falseneg','truepos','falsepos'], path)
    transform_input_data.spirtes_data(*model)

    for prod in itertools.product(list_nsamples, list_ndistributions,list_b,list_d):
        nsamp, ndist, b, d = prod
        if impure_train:
            generate_data_Spirtes.generate_data_multiple_distributions(nsamp, ndist, b, d, linear_train, False)
        else:
            generate_data_Spirtes.generate_data_multiple_distributions(nsamp, ndist, b, d, linear_train)
        # generate 10 files with test distributions.
        if linear_train:
            generate_data_Spirtes.generate_data_linear(nsamp, b, d, test_size, model)
        else:
            generate_data_Spirtes.generate_data_nonlinear(nsamp, b, d, test_size, model)
        train_val = pd.read_csv(
            train_path / 'multiple_distributions_Spirtes_gen_values.csv')
        train_target = pd.read_csv(
            train_path / 'multiple_distributions_Spirtes_gen_targets.csv')
        test_target = pd.read_csv(test_path /
                                  'spirtes_tetrad_constraints_targets.csv')
        for product in list(itertools.product(list_KME, list_K)):
            count += 1
            #print('Count: {}'.format(count))
            KME, K = product
            w = rcc.create_weights(K)

            #print('{} & {}'.format(product, prod))

            x1, y1 = rcc.kernel_mean_embedding(train_val, train_target, w, True, KME)

            result_list = []
            for n in range(test_size):
                test_val = pd.read_csv(test_path / 'spirtes_nonlin_random_b{}_d{}_samples{}_n{}.csv'.format(b,d,nsamp,n))

                result = rcc.kernel_mean_embedding(test_val, test_target, w, False, KME)
                result_list.append(result)


            for E in list_E:

                reg = RFC(n_estimators=E, random_state=0, n_jobs=16).fit(x1, y1)
                #print("RFC fitted")

                for x2, y2 in result_list:

                    prediction = reg.predict(x2)
                    trueneg = 0
                    falseneg = 0
                    truepos = 0
                    falsepos = 0
                    for i in range(len(prediction)):
                        if (prediction[i] == True) and (y2[i] == True):
                            truepos += 1
                        if (prediction[i] == False) and (y2[i] == True):
                            falseneg += 1
                        if (prediction[i] == True) and (y2[i] == False):
                            falsepos += 1
                        if (prediction[i] == False) and (y2[i] == False):
                            trueneg += 1
                    score = reg.score(x2, y2)
                    #print('Score: {}'.format(score))
                    csv.exp_write_csv([linear_train,b,d,KME, E, K, nsamp, ndist, score,trueneg,falseneg,truepos,falsepos],
                              path)
                #print("Finished at time:")
                #print(time.time() - t0)
                #print()


def spirtes_wishart(list_b, list_d, list_b_lin, list_d_lin, list_n_samples, test_size, models, filename,
                    alphas=[0.01]):
    test_target_path = test_path / 'spirtes_tetrad_constraints_targets.csv'
    model_count = 0

    csv.exp_make_csv_predefmodel(['linear','b','d','n_samples','alpha','accuracy','trueneg','falseneg',
                                  'truepos',
                                  'falsepos','model_count'],filename)
    for model in models:
        transform_input_data.spirtes_data(*model)
        targets = pd.read_csv(test_target_path)
        for product in itertools.product(list_n_samples, list_b, list_d):
            n_samples, b, d = product
            generate_data_Spirtes.generate_data_nonlinear(n_samples, b, d, test_size, model)
            for n in range(test_size):
                for alpha in alphas:
                    values = pd.read_csv(test_path / 'spirtes_nonlin_random_b{}_d{}_samples{}_n{}.csv'.format(b, d, n_samples, n))

                    acc, predictions, labels = test_data_single(values, targets, alpha)
                    trueneg = 0
                    falseneg = 0
                    truepos = 0
                    falsepos = 0
                    for i in range(len(labels)):
                        if (labels[i] == True) and (predictions[i] == True):
                            truepos += 1
                        if (labels[i] == False) and (predictions[i] == True):
                            falsepos += 1
                        if (labels[i] == True) and (predictions[i] == False):
                            falseneg += 1
                        if (labels[i] == False) and (predictions[i] == False):
                            trueneg += 1

                    csv.exp_write_csv([False, b, d, n_samples, alpha, acc, trueneg, falseneg, truepos,
                                       falsepos,
                                       model_count],
                                      filename)
        model_count += 1

    #This should test the Wishart test in the linear case.
    for product in itertools.product(list_n_samples, list_b_lin, list_d_lin):
        n_samples, b, d = product
        generate_data_Spirtes.generate_data_linear(n_samples, b, d, test_size, model)
        for n in range(test_size):
            for alpha in alphas:
                values = pd.read_csv(test_path / 'spirtes_random_b{}_d{}_samples{}_n{}.csv'.format(b, d, n_samples, n))

                acc, predictions, labels = test_data_single(values, targets, alpha)
                trueneg = 0
                falseneg = 0
                truepos = 0
                falsepos = 0
                for i in range(len(labels)):
                    if (labels[i] == True) and (predictions[i] == True):
                        truepos += 1
                    if (labels[i] == False) and (predictions[i] == True):
                        falsepos += 1
                    if (labels[i] == True) and (predictions[i] == False):
                        falseneg += 1
                    if (labels[i] == False) and (predictions[i] == False):
                        trueneg += 1

                csv.exp_write_csv([True, b,d, n_samples, alpha, acc, trueneg,falseneg,truepos,falsepos],
                                  filename)


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


def score_kme(linear_train, list_b, list_d, list_E, list_K, list_KME, list_nsamples, test_size,
              list_ndistributions, path):
    kme_path_non_linear = Path('kme_data/non_linear')
    kme_path_linear = Path('kme_data/linear')


    # generate_data_Spirtes.generate_data_multiple_distributions_complex_graph(500, 1000, random, True, model=graph_examples.exampleSpirtes_minimal())

    t0 = time.time()
    count = 0

    csv.exp_make_csv_predefmodel(
        ['train_lin', 'b', 'd', 'KME', 'E', 'K', 'n_samples', 'n_distributions', 'score', 'trueneg',
         'falseneg', 'truepos', 'falsepos'], path)
    for prod in itertools.product(list_nsamples, list_ndistributions, list_b, list_d):
        nsamp, ndist, b, d = prod
        generate_data_Spirtes.generate_data_multiple_distributions(nsamp, ndist, b, d, linear_train)
        train_val = pd.read_csv(
            train_path / 'multiple_distributions_Spirtes_gen_values.csv')
        train_target = pd.read_csv(
            train_path / 'multiple_distributions_Spirtes_gen_targets.csv')

        for product in list(itertools.product(list_KME, list_K)):
            count += 1
            print('Count: {}'.format(count))
            KME, K = product

            print('{} & {}'.format(product, prod))

            kme_filename = 'b{}_d{}_samples{}_K{}_KME{}'.format(b, d, nsamp, K, KME)

            for n in range(test_size):
                x2, y2, w = csv.kme_read_h5(n, kme_path_non_linear, kme_filename)

                x1, y1 = rcc.kernel_mean_embedding(train_val, train_target, w, True, KME)


                for E in list_E:

                    reg = RFC(n_estimators=E, random_state=0, n_jobs=16).fit(x1, y1)
                    print("RFC fitted")

                    prediction = reg.predict(x2)
                    cm = confusion_matrix(y2, prediction)
                    trueneg = cm[0, 0]
                    falseneg = cm[1, 0]
                    truepos = cm[1, 1]
                    falsepos = cm[0, 1]
                    score = reg.score(x2, y2)
                    print('Score: {}'.format(score))
                    csv.exp_write_csv(
                        [linear_train, b, d, KME, E, K, nsamp, ndist, score, trueneg, falseneg, truepos,
                         falsepos],
                        path)
                print("Finished at time:")
                print(time.time() - t0)
                print()

