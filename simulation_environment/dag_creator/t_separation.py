import numpy as np
import networkx as nx

def is_t_separated(graph, A, B):
    # Check if there are sets (C_A, C_B) with #C_A + #C_B < min(#A, #B) that
    # t-separate A and B. (Larger sets C_A, C_B would satisfy the definition
    # of t-separation, but don't lead to conclusions from the theorem, so
    # we don't want to consider those.)
    mL, mO = graph
    d = mL.shape[0]
    minAB = min(np.sum(A), np.sum(B))
    if minAB == 0:
        return False
    # construct a directed graph G such that the directed paths in G correspond
    # to the treks in our graph
    G = nx.DiGraph()
    s = 4 * d
    t = s + 1
    # Other nodes:
    # [0,d), [d,2d): backward part of treks, pre- & post-capacity
    # [2d,3d), [3d,4d): forward part of treks, pre- & post-capacity
    for v in range(d):
        G.add_edge(v, d+v, capacity=1)
        G.add_edge(2*d+v, 3*d+v, capacity=1)
        if A[v]:
            G.add_edge(s, v, capacity=1)
        if B[v]:
            G.add_edge(3*d+v, t, capacity=1)
        for w in range(d):
            if mL[v,w]:
                # v --> w
                G.add_edge(d+w, v, capacity=1)
                G.add_edge(3*d+v, 2*d+w, capacity=1)
            if mO[v,w]:
                # including if v == w
                G.add_edge(d+v, 2*d+w, capacity=1)
    max_flow = nx.maximum_flow_value(G, s, t)
    if max_flow < minAB:
        # there is a cut of size smaller than minAB, so we have a useful t-sep
        # Would it be a possibility to search for the smallest subset
        # t-separating A and B? The size of this subset could serve as an
        # extra label for the ML model.
        return True
    return False

if __name__ == "__main__":
    # Test: construct a graph with six variables that has a t-separation
    # between {2,3} and {4,5}.
    # 0 and 1 with 0 --> 1;
    # 2 and 3 children of 0;
    # 4 and 5 children of 1.
    # (0 and 1 are latent, but the code doesn't need to know that.)
    # Then check if t-separation holds between {2,3} and {4,5}.
    d = 6
    mL = np.zeros((d,d), dtype=bool) # adjacency matrix of directed edges
    mO = np.eye(d, dtype=bool) # adjacency matrix of bidirected edges
    # Add directed edges:
    mL[0,1] = True
    mL[0,2] = mL[0,3] = True
    mL[1,4] = mL[1,5] = True

    graph = (mL, mO)

    # Sets between which we want to check t-separation, encoded as boolean
    # vectors. 6 variables, so the vector is true on the index of the
    # variable it includes.
    A = np.zeros(d, dtype=bool)
    B = np.zeros(d, dtype=bool)
    A[2] = A[3] = True
    B[4] = B[5] = True

    print("is_t_separated(...) =", is_t_separated(graph, A, B))
