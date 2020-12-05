import numpy as np
import pandas as pd
import utility_functions as uf
from scipy.stats import norm

def wishart_test(var0, var1, var2, var3, cov, n):
    det1 = cov.loc[var0,var0] * cov.loc[var3,var3] - cov.loc[var0,var3] * cov.loc[var0,var3]
    det2 = cov.loc[var1,var1] * cov.loc[var2,var2] - cov.loc[var1,var2] * cov.loc[var1,var2]
    product1 = (n + 1) / ((n - 1) * (n - 2)) * det1 * det2
    #product2 = product1 - ((np.linalg.det(cov.loc[[var0,var1,var2,var3],[var0,var1,var2,var3]]) /(n - 2)))
    t = ((3 / (n - 2)) * np.power(np.linalg.det(cov.loc[[var0,var1],[var2,var3]]),2))
    product2 = product1 - (np.linalg.det(cov.loc[[var0,var1,var2,var3],[var0,var1,var2,var3]]) /(n - 2)) + t
    return(np.sqrt(abs(product2)))

def tetrad_test(i, j , k, l, cov, alpha, n):
    test_values = []
    # Might want to add n - 1 / n - 2 before tetrad_ijkl, this is also done in Drton et al (2007)
    tetrad_ijkl = cov.loc[i,j] * cov.loc[k,l] - cov.loc[i,k] * cov.loc[j,l]
    SD = wishart_test(i, j, k, l, cov, n)
    # Probability that a tetrad constraint could attain this value when drawn from a normal distribution.
    test_values.append(2 * (1 - norm.cdf(abs(tetrad_ijkl / SD))))

    tetrad_ijlk = cov.loc[i,j] * cov.loc[k,l] - cov.loc[i,l] * cov.loc[j,k]
    SD = wishart_test(i, j, l, k, cov, n)
    test_values.append(2 * (1 - norm.cdf(abs(tetrad_ijlk / SD))))

    tetrad_iklj = cov.loc[i,k] * cov.loc[j,l] - cov.loc[i,l] * cov.loc[j,k]
    SD = wishart_test(i, k, l, j, cov, n)
    test_values.append(2 * (1 - norm.cdf(abs(tetrad_iklj / SD))))
    # If this returns false, then the probability of seeing the value of x (the p value) is smaller than alpha, which
    # implies that it is not a tetrad constraint.
    return(list(map(lambda x: x > alpha, test_values)))


def test_data(data_path, target_path, b, n_samples):
    alpha = 0.01
    real_values = pd.read_csv(data_path)
    real_cov = real_values.cov()
    real_targets = pd.read_csv(target_path)
    diff = 0
    for i in range(real_targets.shape[0]):
        targets = real_targets.iloc[i,:]
        i1 = 3 * i + 0
        i2 = 3 * i + 1
        i3 = 3 * i + 2
        ilist = [i1, i2, i3]
        labels = []
        for n,ix in enumerate(ilist):
            n = n * 3
            pair1 = targets[n].split()
            pair2 = targets[n + 1].split()
            label = targets[n + 2]

            labels.append(label)
        tetrad = tetrad_test(pair1[0], pair1[1], pair2[0], pair2[1], real_cov, alpha, real_values.shape[0])
        diff += abs((labels).count(True) - (tetrad).count(True))
    print("b and n:")
    print("{} and {}".format(b,n_samples))
    print("The total of differences in t-separations and found tetrad constraints")
    print(diff)
    print("Accuracy:")
    print(1 - (diff / (real_targets.shape[0] * 3)))

def tetrad_test_single(i, j , k, l, cov, alpha, n):
    # Might want to add n - 1 / n - 2 before tetrad_ijkl, this is also done in Drton et al (2007)
    tetrad_ijkl = cov.loc[i,k] * cov.loc[j,l] - cov.loc[i,l] * cov.loc[j,k]
    tetrad_ijkl = ((n - 1) / (n - 2)) *  tetrad_ijkl
    SD = wishart_test(i, j, k, l, cov, n)
    # Probability that a tetrad constraint could attain this value when drawn from a normal distribution.
    test_value = 2 * (1 - norm.cdf(abs(tetrad_ijkl / SD)))

    return(test_value > alpha)

def test_data_single(real_values, real_targets, alpha):
    real_cov = real_values.cov()
    diff = 0
    label_list = []
    tetrad_list = []
    for i in range(real_targets.shape[0]):
        targets = real_targets.iloc[i,:]
        i1 = 3 * i + 0
        i2 = 3 * i + 1
        i3 = 3 * i + 2
        ilist = [i1, i2, i3]
        for n,ix in enumerate(ilist):
            n = n * 3
            pair1 = targets[n].split()
            pair2 = targets[n + 1].split()
            label = targets[n + 2]

            tetrad = tetrad_test_single(pair1[0], pair1[1], pair2[0], pair2[1], real_cov, alpha, real_values.shape[0])

            if label != tetrad:
                diff += 1

            tetrad_list.append(tetrad)
            label_list.append(label)
    accuracy = 1 - (diff / (real_targets.shape[0] * 3))

    return(accuracy, tetrad_list, label_list)


def specific_tests():
    #test_data('real_data/politicaldemocracy.csv', 'real_data/poldem_targets.csv')

    for b in [0, 0.01, 0.02, 0.03, 0.05]:
        for n_samples in [100, 500, 1000]:
            test_data("simulated_data\spirtes_random{}_samples{}.csv".format(b, n_samples),
                      "simulated_data\modified_for_rcc\spirtes_random{}_samples{}_targets.csv".format(b, n_samples), b, n_samples)