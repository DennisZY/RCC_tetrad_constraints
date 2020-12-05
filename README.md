# RCC_tetrad_constraints

This is the code for the experiments in my thesis. The main.py file contains multiple functions that allow for the running of the KME based classifier or Wishart test. Each of these method can be given multiple lists of hyperparamters that will be varied in a grid search. 

- list_b: The non-linear coefficients in the structural model.
- list_d: The non-linear coefficients in the measurement model.
- list_E: The amount of trees in the random forests classifier.
- list_K: The size of the weights for the KME.
- list_KME: The KME embedding, can be set to either 4, minimal or marginal.
- list_nsamples: The amount of samples per random variable.
- list_ndistributions: The amount of sets of samples per training graph.
- alpha: the significance level of the Wishart test.
- impure_train: When set to true, training data is also sampled from graphs that contain an impurity.
- linear_train: When set to true, the data is generated with linear functions. Otherwise it is generated with non-linear functions.

The results will be loaded in a CSV file in the experiment_results folder.
