import graph_examples
import score_functions
import generate_kme_files

def main():
    spirtes_nonlin()

def spirtes_nonlin():
    # Things to vary:
    # embedding
    # Length of KME vector
    # Amount of trees in RFC
    # Amount of non-linearity.

    #model2 = graph_examples.example1()

    linear_train = True
    list_b = [0.1]
    list_d = [0.1]
    list_E = [100]
    list_K = [100]
    list_KME = ['4'] #['minimal', '4', 'marginal']
    list_nsamples = [50,1000]
    test_size = 1
    list_ndistributions = [10]#[200,1000,4000]

    filename = 'test'
    print('building file: {}'.format(filename))
    model = graph_examples.example_spirtes_pure()

    score_functions.spirtes_nonlin(linear_train, list_b, list_d, list_E, list_K, list_KME, list_nsamples,
                                test_size, list_ndistributions, filename, model)


def spirtes_nonlin_impure_train():
    # Things to vary:
    # embedding
    # Length of KME vector
    # Amount of trees in RFC
    # Amount of non-linearity.

    # model2 = graph_examples.example1()

    linear_train = False
    list_b = [0.1]
    list_d = [0.1]
    list_E = [500]
    list_K = [500]
    list_KME = ['4']  # ['minimal', '4', 'marginal']
    list_nsamples = [50]#[50, 100, 500, 1000]
    test_size = 10
    list_ndistributions = [500]#[100, 500, 1000]

    filename = 'experiment_kme_full_graph'
    print('building file: {}'.format(filename))
    model = graph_examples.exampleSpirtes()

    score_functions.spirtes_nonlin(linear_train, list_b, list_d, list_E, list_K, list_KME, list_nsamples,
                                   test_size, list_ndistributions, filename, model, True)


def spirtes_wishart():
    list_b = [0.1]
    list_d = [0.1]
    list_n_samples = [50,100,500,1000]
    test_size = 10
    list_b_lin = [0]
    list_d_lin = [0]
    alpha = [0.001,0.005,0.01,0.05,0.1]

    model = [graph_examples.example_spirtes_pure()]

    filename = 'wishart_experiment_pure_measurement_model'
    print('building file: {}'.format(filename))

    score_functions.spirtes_wishart(list_b, list_d, list_b_lin, list_d_lin, list_n_samples, test_size,
                                    model, filename, alpha)

def parameter_search():
    linear_train = False
    b = 0.1
    d = 0.1
    E = 500
    K = 500
    KME = '4'
    n_samples = 1000
    test_size = 10

    model = graph_examples.exampleSpirtes()

    filename = 'iterative_search_ndistributions'
    print('building file: {}'.format(filename))

    score_functions.iterate_distributions(linear_train, b, d, E, K, KME, n_samples, test_size, filename, model,
                                          True)



if __name__ == "__main__":
    main()