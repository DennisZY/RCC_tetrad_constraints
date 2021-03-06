import itertools

def graph_count():
    return(2)

# All graphs are made with the 24 different permutations of 4 variables,
# even if this does not make any difference for the distribution, it will
# generate the same amount of training examples per graph type.

def example0():
    # Examples with 2 latents and 2 children each
    mv = ['y1','y2','y3','y4']
    models_list = []
    for perm in itertools.permutations(mv,4):
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2}, 'eta2': {y3, y4}}
        s_model = {'eta1': {'eta2'}}
        models_list.append((m_model,s_model))
    return(models_list)

def example0_extra():
    # Examples with 2 latents and 2 children each
    mv = ['y1','y2','y3','y4']
    models_list = []
    for perm in itertools.permutations(mv,4):
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2}, 'eta2': {y3, y4}}
        s_model = {'x1': {'eta2', 'eta1'}}
        models_list.append((m_model,s_model))
    return(models_list)

def example1():
    # 1 latent with 4 children
    mv = ['y1','y2','y3','y4']
    models_list = []
    perms = list(itertools.permutations(mv,4))
    for perm in perms:
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2, y3, y4}}
        s_model = {'eta1': {}}
        models_list.append((m_model,s_model))
    return(models_list)

def example1_extra():
    # 1 latent with 4 children
    mv = ['y1','y2','y3','y4']
    models_list = []
    perms = list(itertools.permutations(mv,4))
    for perm in perms:
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2, y3, y4}}
        s_model = {'x1': {'eta1'}}
        models_list.append((m_model,s_model))
    return(models_list)

def example_intra_construct_impure():
    mv = ['y1','y2','y3','y4']
    models_list = []
    perms = list(itertools.permutations(mv,4))
    for perm in perms:
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2, y3, y4},y1:{y2}}
        s_model = {'eta1': {}}
        models_list.append((m_model,s_model))
    return(models_list)

def example_intra_construct_impure_extra():
    mv = ['y1','y2','y3','y4']
    models_list = []
    perms = list(itertools.permutations(mv,4))
    for perm in perms:
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2, y3, y4},y1:{y2}}
        s_model = {'x1': {'eta1'}}
        models_list.append((m_model,s_model))
    return(models_list)

def example_cross_construct_impure():
    mv = ['y1','y2','y3','y4']
    models_list = []
    for perm in itertools.permutations(mv,4):
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2}, 'eta2': {y3, y4}, y1: {y3}}
        s_model = {'eta1': {'eta2'}}
        models_list.append((m_model,s_model))
    return(models_list)

def example_latent_measure_impure():
    mv = ['y1','y2','y3','y4','y5']
    models_list = []
    for perm in itertools.permutations(mv,5):
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        y5 = perm[4]
        m_model = {'eta1': {y1, y2, y5}, 'eta2': {y3, y4, y5}}
        s_model = {'eta1': {'eta2'}}
        models_list.append((m_model,s_model))
    return(models_list)

def exampleSpirtes():
    s_model = {'L1': {'L2','L3','L4','L5'}}
    m_model = {'L1': {'X1','X2','X3','X4'},
               'L2': {'X5','X6','X7','X8'},
               'L3': {'X9','X10','X11','X12','X8'},
               'L4': {'X13','X14','X15','X16','X12'},
               'L5': {'X17','X18','X19','X20'},
               'X13': {'X9'},
               'X17': {'X18'}}
    return(m_model, s_model)

def example_spirtes_pure():
    s_model = {'L1': {'L2','L3','L4'}}
    m_model = {'L1': {'X1','X2','X3','X4'},
               'L2': {'X5','X6','X7','X8'},
               'L3': {'X9','X10','X11','X12'},
               'L4': {'X13','X14','X15','X16'}}
    return(m_model, s_model)