import numpy as np 

def centers_to_edges(x):
    x = np.asarray(x, dtype=float)

    dx = np.diff(x)

    edges = np.empty(len(x) + 1)

    # internal edges: halfway between neighboring centers
    edges[1:-1] = x[:-1] + dx / 2

    # outer edges: extrapolate half of nearest spacing
    edges[0] = x[0] - dx[0] / 2
    edges[-1] = x[-1] + dx[-1] / 2

    return edges