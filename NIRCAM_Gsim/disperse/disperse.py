from scipy.interpolate import interp1d
import numpy as np
#from polyclip_c import polyclip
from NIRCAM_Gsim.polyclip import polyclip


def dispersed_pixel(x0s,y0s,f0,order,C,ID,oversample_factor=2):
    f = interp1d(f0[0],f0[1],fill_value=0.,bounds_error=False)
    
    #s = interp1d(sens[order][0],sens[order][1],fill_value=0,bounds_error=False)
    s = C.SENS[order]

    x0 = np.mean(x0s)
    y0 = np.mean(y0s)

    dx0s = [t-x0 for t in x0s]
    dy0s = [t-y0 for t in y0s]
    
    # Figuring out a few things about size of order, dispersion and wavelengths to use
    
    #wmin = C.DISPL(order,x0,y0,0.)
    #wmax = C.DISPL(order,x0,y0,1.)
    
    wmin = C.WRANGE[order][0]
    wmax = C.WRANGE[order][1]

    t0 = C.INVDISPL(order,x0,y0,wmin)
    t1 = C.INVDISPL(order,x0,y0,wmax)
    
    dx0 = C.DISPX(order,x0,y0,t0) - C.DISPX(order,x0,y0,t1)
    dx1 = C.DISPY(order,x0,y0,t0) - C.DISPY(order,x0,y0,t1)

    dw = np.abs((wmax-wmin)/(dx1-dx0))
    
    lambdas = np.arange(wmin,wmax+1,np.abs(dw/oversample_factor))
    dS = C.INVDISPL(order,x0,y0,lambdas)

    m = len(lambdas)

    dXs = C.DISPX(order,x0,y0,dS)
    dYs = C.DISPY(order,x0,y0,dS)

    x0s = x0 + dXs 
    y0s = y0 + dYs 

    padding = 1
    l = x0s.astype(np.int32) - padding
    r = x0s.astype(np.int32) + padding
    b = y0s.astype(np.int32) - padding
    t = y0s.astype(np.int32) + padding

    px = np.array([x0s+dx0s[0],x0s+dx0s[1],x0s+dx0s[2],x0s+dx0s[3]],dtype=np.float32).transpose().ravel()
    py = np.array([y0s+dy0s[0],y0s+dy0s[1],y0s+dy0s[2],y0s+dy0s[3]],dtype=np.float32).transpose().ravel()

    lams = np.array([[ll,0,0,0] for ll in lambdas],dtype=np.float32).transpose().ravel()

    poly_inds = np.arange(0,(m+1)*4,4,dtype=np.int32)

    n_poly = len(x0s)

    n = len(lams) # number of pixels we are "dropping", e.g. number of wav bins

    n = n*2
 
    index = np.zeros(n, dtype=np.int32)
    x = np.zeros(n, dtype=np.int32)
    y = np.zeros(n, dtype=np.int32)
    areas = np.zeros(n, dtype=np.float32)
    nclip_poly = np.array([0],np.int32)
    polyclip.polyclip_multi4(l,r,b,t,px,py,n_poly,poly_inds,x,y,nclip_poly,areas,index)


    xs  = x[0:nclip_poly[0]]
    ys  = y[0:nclip_poly[0]]
    areas = areas[0:nclip_poly[0]]
    lams = np.take(lambdas,index)[0:len(xs)]
    counts = f(lams)*areas*s(lams)*np.abs(dw)/oversample_factor

    vg = (xs>=0) & (ys>=0)

    if len(xs[vg])==0:
        return np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),0
    return xs[vg], ys[vg], areas[vg], lams[vg], counts[vg], ID