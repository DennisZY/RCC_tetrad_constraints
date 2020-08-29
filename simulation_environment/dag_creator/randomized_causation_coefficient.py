#ABALONE_X = "/agbs/datasets/causal_pairs_20000/abalone2.data"

import sys
#sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')

import numpy as np
import pandas as pd
import random
from sklearn.ensemble      import RandomForestClassifier as RFC
from scipy.interpolate     import UnivariateSpline as sp
from sklearn.preprocessing import scale, StandardScaler
from sklearn.mixture       import GaussianMixture
from sklearn.metrics       import classification_report, confusion_matrix
import utility_functions

def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),
                    2*np.pi*np.random.rand(k*len(s),1))).T

def f1(x,w):
  return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))

def f2(y1,y2,y3,y4,z):
  return np.hstack((f1(y1,wy1).mean(0),f1(y2,wy2).mean(0),f1(y3,wy3).mean(0),
                    f1(y4,wy4).mean(0),
                    f1(z,wz).mean(0)))
def f2_extra(y1,y2,y3,y4,p1,p2,p3,p4,p5,p6,z):
  return np.hstack((f1(y1,wy1).mean(0),f1(y2,wy2).mean(0),f1(y3,wy3).mean(0),
                    f1(y4,wy4).mean(0),
                    f1(p1, wp).mean(0),
                    f1(p2, wp).mean(0),
                    f1(p3, wp).mean(0),
                    f1(p4, wp).mean(0),
                    f1(p5, wp).mean(0),
                    f1(p6, wp).mean(0),
                    f1(z,wz).mean(0)))

def f2_cov(y1,y2,y3,y4):
    return(f1(cov_array(y1,y2,y3,y4), wc).mean(0))

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

# Return a dictionary with (variable,data) key value pairs.
def get_values(values_df, i):
    # Generate data and label from a randomly chosen graph.
    values = {}
    # Amount of each y in a row.
    vl = round(values_df.shape[1] / 4)
    # TODO y1 until y4 are hardcoded by their names, should vary based on
    #  the names of the measure variables that I give to the function.
    values['y1'] = np.array(values_df.iloc[i][0:vl]).reshape(vl, 1)
    values['y2'] = np.array(values_df.iloc[i][vl:vl * 2]).reshape(vl, 1)
    values['y3'] = np.array(values_df.iloc[i][vl * 2:vl * 3]).reshape(vl, 1)
    values['y4'] = np.array(values_df.iloc[i][vl * 3:vl * 4]).reshape(vl, 1)
    return(values)

# Generate (S, l) combinations, where S is the kernel mean embedding
def kernel_mean_embedding(values_df, targets_df):
    # The second argument below is the amount of columns, should be based on
    # the # of features in the kernel mean embedding, which is the amount of
    kernel_features = 11
    Z1 = np.zeros((3 * values_df.shape[0],kernel_features*wy1.shape[1]))
    L1 = np.ones(3 * values_df.shape[0])
    for i in range(values_df.shape[0]):
        values = get_values(values_df, i)

        targets = targets_df.iloc[i]

        i1       = 3*i+0
        i2       = 3*i+1
        i3       = 3*i+2
        ilist = [i1,i2,i3]
        # Here the kernel mean embeddings are added to a list.
        # Select row i1 from Z1 and all columns
        for n,ix in enumerate(ilist):
            n = n * 3
            pair1 = targets[n].split()
            pair2 = targets[n + 1].split()
            y1 = values[pair1[0]]
            y2 = values[pair1[1]]
            y3 = values[pair2[0]]
            y4 = values[pair2[1]]
            label = targets[n + 2]
            Z1[ix,:] = f2_extra(y1,y2,y3,y4,
                          np.hstack((y1,y2)),
                          np.hstack((y3,y4)),
                          np.hstack((y1, y3)),
                          np.hstack((y2, y4)),
                          np.hstack((y1, y4)),
                          np.hstack((y2, y3)),
                          np.hstack((y1,y2,y3,y4)))

            # Z1[ix,:] = f2(y1,y2,y3,y4,np.hstack((y1,y2,y3,y4)))

            #Z1[ix,:] = f2_cov(y1,y2,y3,y4)

            L1[ix] = label

        # Print i after 100 steps.
        if(np.mod(i,100)==0 and i > 500):
            print(i)

    return (Z1,L1.ravel())

def normal_embedding(values_df, targets_df):
    Z1 = np.zeros((3 * values_df.shape[0], 10))
    #Z1 = np.zeros((3 * values_df.shape[0], values_df.shape[1]))

    L1 = np.ones(3 * values_df.shape[0])
    for i in range(values_df.shape[0]):
        values = get_values(values_df, i)

        targets = targets_df.iloc[i]

        i1 = 3 * i + 0
        i2 = 3 * i + 1
        i3 = 3 * i + 2
        ilist = [i1, i2, i3]
        # Here the kernel mean embeddings are added to a list.
        # Select row i1 from Z1 and all columns
        for n, ix in enumerate(ilist):
            n = n * 3
            pair1 = targets[n].split()
            pair2 = targets[n + 1].split()
            y1 = values[pair1[0]]
            y2 = values[pair1[1]]
            y3 = values[pair2[0]]
            y4 = values[pair2[1]]
            label = targets[n + 2]

            Z1[ix, :] = cov_array(y1,y2,y3,y4).flatten()

            #Z1[ix, :] = np.vstack((y1,y2,y3,y4)).flatten()

            L1[ix] = label

        # Print i after 100 steps.
        if (np.mod(i, 100) == 0 and i > 500):
            print(i)

    return (Z1, L1.ravel())

np.random.seed(0)

N = 3000
K = 100
E = 1000

values = pd.read_csv('generated_data.nosync/gen_values.csv')
targets = pd.read_csv('generated_data.nosync/gen_targets.csv')

randomsamples = values.shape[0]

randomlist = random.sample(list(range(values.shape[0])), randomsamples)

random_values = values.iloc[randomlist]
random_targets = targets.iloc[randomlist]

def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),
                    2*np.pi*np.random.rand(k*len(s),1))).T

#shape = (2,300)
wy1 = rp(K,[0.15,1.5,15],1) # 2.60, 0.78
wy2 = rp(K,[0.15,1.5,15],1) # 2.60, 0.78
wy3 = rp(K,[0.15,1.5,15],1) # 2.60, 0.78
wy4 = rp(K,[0.15,1.5,15],1) # 2.60, 0.78
#shape = (4,300)
wz = rp(K,[0.08,0.8,8],4)
wp = rp(K, [0.08, 0.8, 8], 2)
wc = rp(K, [0.08, 0.8, 8], 1)

# x1 = S, y1 = l
# Replace with my own data.
#(x1,y1) = kernel_mean_embedding(random_values, random_targets)
(x1,y1) = kernel_mean_embedding(random_values, random_targets)

print("UNIQUE COUNTS")
print(np.unique(y1, return_counts=True))

reg  = RFC(n_estimators=E,random_state=0,n_jobs=16).fit(x1,y1);

for i in range(utility_functions.get_graph_count()):
    print("Graph {} score.".format(i))
    test_values = pd.read_csv('generated_data.nosync/graph{'
                              '}_gen_values.csv'.format(i))
    test_targets = pd.read_csv('generated_data.nosync/graph{}_gen_targets.csv'.format(i))
    (x1_test, y1_test) = kernel_mean_embedding(test_values, test_targets)
    prediction = reg.predict(x1_test)
    print(reg.score(x1_test, y1_test))
    print("Unique counts of graph {}".format(i))
    print(np.unique(y1_test, return_counts=True))
    #print(classification_report(y1_test, prediction))
    print(confusion_matrix(y1_test, prediction))


# (R1,R2,R3,M) = abalone(reg)
# M1 = (M>np.percentile(M,10))*M
# M2 = (M>np.percentile(M,20))*M
# M3 = (M>np.percentile(M,30))*M
# M4 = (M>np.percentile(M,40))*M
# M5 = (M>np.percentile(M,50))*M
# M6 = (M>np.percentile(M,60))*M
# M7 = (M>np.percentile(M,70))*M
# M8 = (M>np.percentile(M,80))*M
# M9 = (M>np.percentile(M,90))*M
#
# import networkx as nx
# import matplotlib.pyplot as plt

# def plot(M):
#   G=nx.from_numpy_matrix(M,create_using=nx.MultiDiGraph())
#
#   labels={}
#   labels[0]=r'LEN'
#   labels[1]=r'DIA'
#   labels[2]=r'HEI'
#   labels[3]=r'WEI'
#   labels[4]=r'WEA'
#   labels[5]=r'WEB'
#   labels[6]=r'WEC'
#   labels[7]=r'RIN'
#
#   pos=nx.circular_layout(G)
#   nx.draw_networkx_nodes(G,pos,node_size=1000,node_color='white')
#   nx.draw_networkx_edges(G,pos)
#   nx.draw_networkx_labels(G,pos,labels,font_size=12)
#   print nx.is_directed_acyclic_graph(G)
#   print labels
#   print G.edges()
#   plt.show()

#plot(M5)
