import itertools

def graph_count():
    return(3)

# All graphs are made with the 24 different permutations of 4 variables,
# even if this does not make any difference for the distribution, it will
# generate the same amount of training examples per graph type.

def example3():
    # Examples with 4 connected latent variables
    mv = ['y1','y2','y3','y4']
    models_list = []
    for perm in itertools.permutations(mv,4):
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1}, y1: {y2, y3, y4}, y2: {y3, y4},
                   y3: {y4}}
        s_model = {}
        models_list.append((m_model,s_model))
    return(models_list)

def example0():
    # Different example of no t-sep.
    mv = ['y1','y2','y3','y4']
    models_list = []
    for perm in itertools.permutations(mv,4):
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1}, y1: {y2, y3}, y2: {y4},y3: {y4}}
        s_model = {}
        models_list.append((m_model,s_model))
    return(models_list)

def example1():
    # Examples with 2 latents and 2 children each
    mv = ['y1','y2','y3','y4']
    models_list = []
    for perm in itertools.permutations(mv,4):
        y1 = perm[0]
        y2 = perm[1]
        y3 = perm[2]
        y4 = perm[3]
        m_model = {'eta1': {y1, y2}, 'eta2': {y3, y4}}
        s_model = {'x1': {'eta2'}, 'eta1': {'eta2'}}
        #s_model = {'eta1': {'eta2'}}
        models_list.append((m_model,s_model))
    return(models_list)

def example2():
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