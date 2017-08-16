import numpy as np
import grismconf
import os
from astropy.io import fits
from scipy import sparse
from astropy.table import Table
#from NIRCAM_Gsim import polyclip
from NIRCAM_Gsim.disperse import *

def helper(vars):
    x0s,y0s,f,order,C,ID = vars # in this case ID is dummy number
    p = dispersed_pixel(x0s,y0s,f,order,C,ID)
    xs, ys, areas, lams, counts,ID = p
    IDs = [ID] * len(xs)

    pp = np.array([xs, ys, areas, lams, counts,IDs])
    return pp

class observation():
    # This class defines an actual observations. It is tied to a single flt and a single config file
    
    def __init__(self,direct_images,segmentation_data,config,cross_filter=None,mod="A",order="+1",plot=0,max_split=100,SED_file=None):
        """direct_images: List of file name containing direct imaging data
        segmentation_data: an array of the size of the direct images, containing 0 and 1's, 0 being pixels to ignore
        config: The path and name of a GRISMCONF NIRCAM configuration file
        passband: The name of a direct filter passband file, ascii, 1st column being wavelength in A and second the throughput
        mod: Module, A or B
        order: The name of the spectral order to simulate, +1 or +2 for NIRCAM
        max_split: Number of chunks to compute instead of trying everything at once.
        SED_file: Name of HDF5 file containing datasets matching the ID in the segmentation file and each consisting of a [[lambda],[flux]] array.
        """

        self.C = grismconf.Config(config)
        if self.C.__version__!=1.2:
            print "Need grismconf v.1.2"
            sys.exit(-1)
            
        if plot:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.clf()
            x = np.arange(self.C.WMIN,self.C.WMAX,(self.C.WMAX,self.C.WMIN)/100.)
            plt.plot(x,self.C.SENS[order](x))

        self.dir_images = direct_images
        self.seg = segmentation_data
        self.dims = np.shape(self.seg)
        self.order = order
        self.SED_file = SED_file

        self.create_pixel_list()
        
        self.p_l = []
        self.p_a = []


        self.minx = int(min(self.xs))
        self.maxx= int(max(self.xs))
        self.miny = int(min(self.ys))
        self.maxy = int(max(self.ys))


        print "Splitting in chunks of",max_split
        self.xs = np.array_split(self.xs,max_split)
        self.ys = np.array_split(self.ys,max_split)
        for l in self.fs.keys():
            self.fs[l] = np.array_split(self.fs[l],max_split)

    def create_pixel_list(self):

        self.ys,self.xs = np.nonzero(self.seg)

        print len(self.xs),"pixels to process"
        self.fs = {}
        for dir_image in self.dir_images:
            if self.SED_file==None:
                try:
                    l = fits.getval(dir_image,'PHOTPLAM') / 10000. # in Angsrrom and we want Micron now
                except:
                    print("WARNING: unable to find PHOTPLAM keyword in {}".format(dir_image))
                    sys.exit()

                try:
                    photflam = fits.getval(dir_image,'photflam')
                except:
                    print("WARNING: unable to find PHOTFLAM keyword in {}".format(dir_image))
                    sys.exit()
                print("Loaded",dir_image, "wavelength:",l,"A")
            try:
                d = fits.open(dir_image)[1].data
            except:
                d = fits.open(dir_image)[0].data

            # If we do not use an SED file then we use photometry to get fluxes
            # Otherwise, we assume that objects are normalized to 1.
            if self.SED_file==None:
                self.fs[l] = d[self.ys,self.xs] * photflam
            else:
                self.fs["SED"] = d[self.ys,self.xs]

    def disperse_all(self):
        self.simulated_image = np.zeros(self.dims,np.float)
        for c in range(len(self.xs)):
            print c+1,"of",len(self.xs)
            self.disperse_chunk(c)
            #import nf
            #nf.disp(self.simulated_image,2)

    def disperse_chunk(self,c):
        """Method that handles the dispersion. To be called after create_pixel_list()"""
        from multiprocessing import Pool
        from progressbar import Bar, ETA, ReverseBar, ProgressBar, Percentage
        import time

        if self.SED_file!=None:
            import h5py
            h5f = h5py.File(self.SED_file,'r')
            # b = h5f['16524'][:]
            pars = []
            for i in range(len(self.xs[c])):
                ID = int(self.seg[self.ys[c][i],self.xs[c][i]])
                tmp = h5f["%s" % (ID)][:]
                lams = tmp[0]
                fffs = tmp[1]*self.fs["SED"][c][i]
                f = [lams,fffs]

                xs0 = [self.xs[c][i],self.xs[c][i]+1,self.xs[c][i]+1,self.xs[c][i]]
                ys0 = [self.ys[c][i],self.ys[c][i],self.ys[c][i]+1,self.ys[c][i]+1]
                pars.append([xs0,ys0,f,self.order,self.C,ID])
            h5f.close()
        #sys.exit(1)
        # test...
        # def Gaussian1D(x,A,mu,sigma):
        #     return A/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
        # lams = np.arange(20000.,50000,5)
        # fffs = Gaussian1D(lams,1e-17,35000,5)+Gaussian1D(lams,2e-17,37000,5)+Gaussian1D(lams,3e-17,36000,5)
        # f = [lams,fffs]
        # pars = []
        # for i in range(len(self.xs[c])):
        #     ID = i
        #     xs0 = [self.xs[c][i],self.xs[c][i]+1,self.xs[c][i]+1,self.xs[c][i]]
        #     ys0 = [self.ys[c][i],self.ys[c][i],self.ys[c][i]+1,self.ys[c][i]+1]
        #     pars.append([xs0,ys0,f,self.order,self.C,ID])
        #

        else:
        # good code below
            pars = []
            for i in range(len(self.xs[c])):
                ID = i
                xs0 = [self.xs[c][i],self.xs[c][i]+1,self.xs[c][i]+1,self.xs[c][i]]
                ys0 = [self.ys[c][i],self.ys[c][i],self.ys[c][i]+1,self.ys[c][i]+1]
                lams = self.fs.keys()
                f = [lams,[self.fs[l][c][i] for l in self.fs.keys()]]
                pars.append([xs0,ys0,f,self.order,self.C,ID])

        print len(pars),"pixels loaded for dispersion..."
        
        time1 = time.time()
        mypool = Pool(10) # Create pool
        all_res = mypool.imap_unordered(helper,pars) # Stuff the pool
        mypool.close() # No more work

        widgets=[Percentage(), Bar(), ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(pars)).start()

        #simulated_image = np.zeros(self.dims,np.float)
        for i,pp in enumerate(all_res, 1):        
            if np.shape(pp.transpose())==(1,6):
                continue
            #print np.min(pp[0]),np.max(pp[0])
            x,y,f = pp[0],pp[1],pp[4]

            vg = (x>=0) & (x<self.dims[1]) & (y>=0) & (y<self.dims[0]) 

            x = x[vg]
            y = y[vg]
            f = f[vg]
            
            if len(x)<1:
                continue

            minx = int(min(x))
            maxx= int(max(x))
            miny = int(min(y))
            maxy = int(max(y))

            a = sparse.coo_matrix((f, (y-miny, x-minx)), shape=(maxy-miny+1,maxx-minx+1)).toarray()
            self.simulated_image[miny:maxy+1,minx:maxx+1] = self.simulated_image[miny:maxy+1,minx:maxx+1] + a
            
            if i % len(pars)/100:
                pbar.update(i)
        pbar.finish()
        time2 = time.time()

        print time2-time1,"s."
        #return simulated_image

    def show(self):
        import matplotlib.pyplot as plt
        plt.ion()

        xx = self.p_x - min(self.p_x)
        yy = self.p_y - min(self.p_y)
        a = sparse.coo_matrix((self.p_f, (yy, xx)), shape=(max(yy)+1, max(xx)+1)).toarray()

        im = plt.imshow(a)
        im.set_clim(0,1)

        plt.draw()
        raw_input("...")

