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

    models = graph_examples.example0()
    model = models[0]
    #model2 = graph_examples.example1()

    linear_train = [False]
    list_b = [0.01,0.05]
    list_d = [0.01,0.05]
    filename = 'experiment_1_t_sep'
    list_E = [100,500,1000]
    list_K = [100,500,1000]
    list_KME = ['4'] #['minimal', '4', 'marginal']
    list_nsamples = [50,100,500,1000]
    test_size = 100
    list_ndistributions = [100,500,1000]#[200,1000,4000]

    print('building file: {}'.format(filename))


    score_functions.spirtes_nonlin(linear_train, list_b, list_d, list_E, list_K, list_KME, list_nsamples,
                              test_size, list_ndistributions, filename, model)

def spirtes_wishart():
    models = graph_examples.example_latent_measure_impure()
    model = [models[0]]
    list_b = [0.01,0.05]
    list_d = [0.01,0.05]
    list_n_samples = [50,100,500,1000]
    test_size = 100
    filename = 'wishart_experiment_latent_measure_impure'
    print('building file: {}'.format(filename))
    list_b_lin = [0, 0.1, 0.05]
    list_d_lin = [0]

    score_functions.spirtes_wishart(list_b, list_d, list_b_lin, list_d_lin, list_n_samples, test_size,
                                    model, filename)


def score_kme():
    linear_train = [False]
    list_b = [0.05]
    list_d = [0.05]
    path = 'experiment_general'
    list_E = [500]
    list_K = [500]
    list_KME = ['4']  # ['minimal', '4', 'marginal']
    list_nsamples = [100]
    test_size = 1
    list_ndistributions = [1000]  # [200,1000,4000]

    score_functions.score_kme(linear_train, list_b, list_d, list_E, list_K, list_KME, list_nsamples,
                              test_size, list_ndistributions, path)

def create_kme():
    model = graph_examples.exampleSpirtes()

    list_b = [0.1]
    list_d = [0.01, 0.1]
    list_K = [100, 500, 1000]
    list_KME = ['4']  # ['minimal', '4', 'marginal']
    list_nsamples = [100, 1000, 10000]
    test_size = 10

    generate_kme_files.generate_KME(list_b, list_d, list_K, list_KME, list_nsamples, test_size, model)


if __name__ == "__main__":
    main()