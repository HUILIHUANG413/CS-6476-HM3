import numpy as np
def normalizeHomogeneous(points):
    #[u * w, v * w, w] to[u, v, 1]
    npoints=points.shape[1]
    r,c=points.shape
    normalizedPoints=np.zeros((r,c))
    for i in range (0,npoints):
        normalizedPoints[:,i]=points[:,i]/points[2,i]
    return normalizedPoints