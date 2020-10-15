#ABALONE_X = "/agbs/datasets/causal_pairs_20000/abalone2.data"

import sys
#sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')

import itertools
import numpy as np
import pandas as pd
import random
import utility_functions as uf
from sklearn.metrics import confusion_matrix, classification_report
import statistics
import csv_functions as csv


def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),
                    2*np.pi*np.random.rand(k*len(s),1))).T

def f1(x,w):
  return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))

def f2(y1,y2,y3,y4,z,w,wz):
  return np.hstack((f1(y1,w[0]).mean(0),f1(y2,w[1]).mean(0),f1(y3,w[2]).mean(0),
                    f1(y4,w[3]).mean(0),
                    f1(z,wz).mean(0)))

def f2_extra(y1,y2,y3,y4,p1,p2,z, w, wz, wp):
  return np.hstack((f1(y1,w[0]).mean(0),f1(y2,w[1]).mean(0),f1(y3,w[2]).mean(0),
                    f1(y4,w[3]).mean(0),
                    f1(p1, wp).mean(0),
                    f1(p2, wp).mean(0),
                    f1(z,wz).mean(0)))

def f2_minimal(p1,p2,wp1,wp2):
  return np.hstack((f1(p1, wp1).mean(0),
                    f1(p2, wp2).mean(0)))

def f2_4(z, wz):
  return np.hstack((f1(z,wz).mean(0)))

def f2_marg(y1, y2, y3, y4, w):
    return np.hstack((f1(y1, w[0]).mean(0), f1(y2, w[1]).mean(0), f1(y3, w[2]).mean(0),
                      f1(y4, w[3]).mean(0)))

def f2_cov(y1,y2,y3,y4):
    return(f1(cov_array(y1,y2,y3,y4), wc).mean(0))


# Returns a covariance array of four variables, to be used as feature for a classifier.
def cov_array(y1,y2,y3,y4):
    cov_matrix = pd.DataFrame((y1.flatten(), y2.flatten(), y3.flatten(),
                               y4.flatten())).T.cov()
    cov_array = []
    for i in range(cov_matrix.shape[0]):
        for j in range(i, cov_matrix.shape[0]):
            if i == j:
                cov_array.append(cov_matrix.iloc[i, j])
            else:
                cov_array.append(np.sqrt(2) * (cov_matrix.iloc[i, j]))
    return(np.array(cov_array).reshape(len(cov_array),1))

def extract_value_df(df, var):
    val_shape = df.shape[0]
    y1 = np.array(df[var[0]]).reshape(val_shape, 1)
    y2 = np.array(df[var[1]]).reshape(val_shape, 1)
    y3 = np.array(df[var[2]]).reshape(val_shape, 1)
    y4 = np.array(df[var[3]]).reshape(val_shape, 1)
    return(y1, y2, y3, y4)

# Generate (S, l) combinations, where S is the kernel mean embedding
def kernel_mean_embedding(values, targets, weights, train = False, case = 'minimal'):
    # The second argument below is the amount of columns, should be based on
    # the # of features in the kernel mean embedding, which is the amount of
    if case == '4': kernel_features = 1
    elif case == 'minimal': kernel_features = 2
    elif case == 'marginal': kernel_features = 4
    elif case == 'normal': kernel_features = 5
    elif case == 'extra': kernel_features = 7

    print('{} rows to embed'.format(targets.shape[0]))

    w, wz, wp, wp1, wp2 = weights
    L1 = np.zeros((3 * targets.shape[0], 1))
    Z1 = np.zeros((3 * targets.shape[0], kernel_features * w[0].shape[1]))
    for i in range(0,targets.shape[0]):
        target_str = targets.iloc[i]
        if train:
            p1 = target_str[0].split()
            p2 = target_str[1].split()
            values_df = uf.get_values(values, i, [p1[0], p1[1], p2[0], p2[1]])
        else:
            values_df = values
        i1 = 3 * i + 0
        i2 = 3 * i + 1
        i3 = 3 * i + 2
        ilist = [i1, i2, i3]
        # Here the kernel mean embeddings are added to a list.
        # Select row i1 from Z1 and all columns
        for n, ix in enumerate(ilist):
            n = n * 3
            pair1 = target_str[n].split()
            pair2 = target_str[n + 1].split()
            y1, y2, y3, y4 = extract_value_df(values_df, [pair1[0],pair1[1],pair2[0],pair2[1]])
            label = target_str[n + 2]
            if case == 'minimal':
                Z1[ix, :] = f2_minimal(np.hstack((y1, y2)), np.hstack((y3, y4)),wp1, wp2)
            elif case == 'extra':
                Z1[ix,:] = f2_extra(y1,y2,y3,y4,
                              np.hstack((y1,y2)),
                              np.hstack((y3,y4)),
                              np.hstack((y1,y2,y3,y4)),
                              w,
                              wz,
                              wp)
            elif case == '4':
                Z1[ix,:] = f2_4(np.hstack((y1,y2,y3,y4)), wz)

            elif case == 'normal':
                Z1[ix, :] = f2(y1, y2, y3, y4, np.hstack((y1, y2, y3, y4)), w, wz)

            elif case == 'marginal':
                Z1[ix, :] = f2_marg(y1, y2, y3, y4, w)

            L1[ix] = label

        if i % 1000 == 0:
            print(i)

    return (Z1,L1.ravel())


# Variant of the kernel mean embedding function where only the kernel mean embedding of every tetrad is computed.
def kernel_mean_embedding_nolabel(values, w, wz, wp, train):
    kernel_features = 5
    if train:
        Z1 = np.zeros((len(values), kernel_features * w[0].shape[1]))
        for i, tetrad_df in enumerate(values):
            y1, y2, y3, y4 = extract_value_df(tetrad_df, list(tetrad_df.columns))
            Z1[i, :] = f2(y1, y2, y3, y4, np.hstack((y1, y2, y3, y4)), w, wz)
    else:
        tetrads = list(itertools.combinations(list(values.columns), 4))
        Z1 = np.zeros((len(tetrads),kernel_features*w[0].shape[1]))
        for i, t in enumerate(tetrads):
            y1, y2, y3, y4 = extract_value_df(values, t)
            Z1[i, :] = f2(y1, y2, y3, y4, np.hstack((y1, y2, y3, y4)), w, wz)
    return (Z1)

def median_heuristic(df):
    row_len = len(df.index)
    distances_list = []
    for i in range(0,row_len):
        for j in range(i + 1, row_len):
            # calculate the square of the norm, which is just the sum of every element squared.
            distances_list.append(np.sum(np.square(df.iloc[i,:] - df.iloc[j,:])))
    distances_list.sort()
    return(np.sqrt(statistics.median(distances_list) / 2))


def compare_distributions(values, train=False):
    w, wz, wp = create_weights(400)
    x1 = kernel_mean_embedding_nolabel(values, w, wz, wp, train)
    return(x1)


def create_weights(K):
    np.random.seed(0)
    # shape = (2,300)
    w1 = rp(K, [0.4, 4, 40], 1)  # 2.60, 0.78
    w2 = rp(K, [0.4, 4, 40], 1)
    w3 = rp(K, [0.4, 4, 40], 1)
    w4 = rp(K, [0.4, 4, 40], 1)

    # shape = (4,300)
    wz = rp(K, [0.2, 2, 20], 4)
    wp = rp(K, [0.2, 2, 20], 2)

    wp1 = rp(K, [0.2, 2, 20], 2)
    wp2 = rp(K, [0.2, 2, 20], 2)

    return([w1,w2,w3,w4], wz, wp, wp1, wp2)

def main():
    print()

if __name__ == "__main__":
    main()
