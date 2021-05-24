

def helper(vars):
    from ..disperse.disperse import dispersed_pixel
    import numpy as np

    x0s,y0s,f,order,C,ID,extrapolate_SED, xoffset, yoffset = vars # in this case ID is dummy number
    p = dispersed_pixel(x0s,y0s,f,order,C,ID,extrapolate_SED=extrapolate_SED,xoffset=xoffset,yoffset=yoffset)
    xs, ys, areas, lams, counts,ID = p
    IDs = [ID] * len(xs)

    pp = np.array([xs, ys, areas, lams, counts,IDs])
    return pp

