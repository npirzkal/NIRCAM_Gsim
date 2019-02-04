import numpy as np
import grismconf
import os
from astropy.io import fits
from scipy import sparse
from astropy.table import Table
#from NIRCAM_Gsim import polyclip
#from NIRCAM_Gsim.disperse import *
from ..disperse.disperse import dispersed_pixel
import h5py

def helper(vars):
    x0s,y0s,f,order,C,ID,extrapolate_SED = vars # in this case ID is dummy number
    p = dispersed_pixel(x0s,y0s,f,order,C,ID,extrapolate_SED=extrapolate_SED)
    xs, ys, areas, lams, counts,ID = p
    IDs = [ID] * len(xs)

    pp = np.array([xs, ys, areas, lams, counts,IDs])
    return pp

class observation():
    # This class defines an actual observations. It is tied to a single flt and a single config file
    
    def __init__(self,direct_images,segmentation_data,config,mod="A",order="+1",plot=0,max_split=100,SED_file=None,extrapolate_SED=False,max_cpu=10,ID=0,SBE_save=None, boundaries=[]):
        """direct_images: List of file name containing direct imaging data
        segmentation_data: an array of the size of the direct images, containing 0 and 1's, 0 being pixels to ignore
        config: The path and name of a GRISMCONF NIRCAM configuration file
        mod: Module, A or B
        order: The name of the spectral order to simulate, +1 or +2 for NIRCAM
        max_split: Number of chunks to compute instead of trying everything at once.
        SED_file: Name of HDF5 file containing datasets matching the ID in the segmentation file and each consisting of a [[lambda],[flux]] array.
        SBE_save: If set to a path, HDF5 containing simulated stamps for all obsjects will be saved.
        boundaries: a tuple containing the coordinates of the FOV within the larger seed image. Needs to be specified if SBE_save!=None
        """

        self.C = grismconf.Config(config)
        if self.C.__version__!=1.2:
            print("Need grismconf v.1.2")
            sys.exit(-1)
            
        if plot:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.clf()
            x = np.arange(self.C.WMIN,self.C.WMAX,(self.C.WMAX,self.C.WMIN)/100.)
            plt.plot(x,self.C.SENS[order](x))

        self.ID = ID
        self.dir_images = direct_images
        self.seg = segmentation_data
        self.dims = np.shape(self.seg)
        self.order = order
        self.SED_file = SED_file
        self.SBE_save = SBE_save
        self.max_cpu = max_cpu

        if SBE_save!=None:
            if len(boundaries)!=4:
                print("WARMING: boundaries needs to be specficied if using SBE_save")
                sys.exit()
            self.xstart,self.xend,self.ystart,self.yend = boundaries

        self.extrapolate_SED = extrapolate_SED # Allow for SED extrapolation
        if self.extrapolate_SED:
            print("Warning: SED Extrapolation turned on.")

        self.create_pixel_list()
        
        self.p_l = []
        self.p_a = []



    #def create_pixel_list_by_ID(self):
    def create_pixel_list(self):
        # Create a list of pixels to dispersed, grouped per object ID
        if self.ID==0:
            self.xs = []
            self.ys = []
            all_IDs = np.array(list(set(np.ravel(self.seg))))
            all_IDs = all_IDs[all_IDs>0]
            print("We have ",len(all_IDs),"Objects")
            for ID in all_IDs:
                ys,xs = np.nonzero(self.seg==ID)
                self.xs.append(xs)
                self.ys.append(ys)
            self.IDs = all_IDs
        else:
            vg = self.seg==self.ID
            ys,xs = np.nonzero(vg)            
            self.xs.append(xs)
            self.ys.append(ys)
            self.IDs = [self.ID]

        self.fs = {}
        for dir_image in self.dir_images:
            print("dir image:",dir_image)
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
                print("Loaded",dir_image, "wavelength:",l,"micron")
            try:
                d = fits.open(dir_image)[1].data
            except:
                d = fits.open(dir_image)[0].data


            # If we do not use an SED file then we use photometry to get fluxes
            # Otherwise, we assume that objects are normalized to 1.
            if self.SED_file==None:
                self.fs[l] = []
                for i in range(len(self.IDs)):
                    self.fs[l].append(d[self.ys[i],self.xs[i]] * photflam)
            else:
                # Need to normalize the object stamps              
                for ID in self.IDs:
                    vg = self.seg==ID
                    sum_seg = np.sum(d[vg])
                    if sum_seg!=0.:
                        d[vg] = d[vg]/sum_seg

                self.fs["SED"] = []
                for i in range(len(self.IDs)):
                    self.fs["SED"].append(d[self.ys[i],self.xs[i]])



    # def create_pixel_listOLD(self):

    #     if self.ID==0:
    #         self.ys,self.xs = np.nonzero(self.seg)
    #     else:
    #         #print("looking at",self.ID)
    #         vg = self.seg==self.ID
    #         self.ys,self.xs = np.nonzero(vg)            
    #         #print self.ys,self.xs

    #     print(len(self.xs),"pixels to process")
    #     self.fs = {}
    #     for dir_image in self.dir_images:
    #         print("dir image:",dir_image)
    #         if self.SED_file==None:
    #             try:
    #                 l = fits.getval(dir_image,'PHOTPLAM') / 10000. # in Angsrrom and we want Micron now
    #             except:
    #                 print("WARNING: unable to find PHOTPLAM keyword in {}".format(dir_image))
    #                 sys.exit()

    #             try:
    #                 photflam = fits.getval(dir_image,'photflam')
    #             except:
    #                 print("WARNING: unable to find PHOTFLAM keyword in {}".format(dir_image))
    #                 sys.exit()
    #             print("Loaded",dir_image, "wavelength:",l,"micron")
    #         try:
    #             d = fits.open(dir_image)[1].data
    #         except:
    #             d = fits.open(dir_image)[0].data

    #         # If we do not use an SED file then we use photometry to get fluxes
    #         # Otherwise, we assume that objects are normalized to 1.
    #         if self.SED_file==None:
    #             self.fs[l] = d[self.ys,self.xs] * photflam
    #         else:
    #             # Need to normalize the object stamps
    #             IDs = set(np.ravel(self.seg))
                
    #             for ID in IDs:
    #                 if ID==0: continue
    #                 vg = self.seg==ID
    #                 sum_seg = np.sum(d[vg])
    #                 if sum_seg!=0.:
    #                     d[vg] = d[vg]/sum_seg
    #             #print "we have:",len(IDs)
    #             self.fs["SED"] = d[self.ys,self.xs]
    #             #print "sum of this object:",np.sum(self.fs["SED"])

    def disperse_all(self):
        self.simulated_image = np.zeros(self.dims,np.float)

        if self.SBE_save != None:
            print("Outputing to ", self.SBE_save)

            #if os.path.isfile(self.SBE_save):
            #    os.unlink(self.SBE_save)
            fhdf5 = h5py.File(self.SBE_save,"a")

        for i in range(len(self.IDs)):
            print("Dispersing ",i+1,"of",len(self.IDs),"ID:",self.IDs[i])
            this_object = self.disperse_chunk(i)

            if self.SBE_save != None:
                # If SBE_save is enabled, we create an HDF5 file containing the stamp of this simulated object
                # order is in self.order
                # We just save the x,y,f,w arrays as well as info about minx,maxx,miny,maxy
                

                # We trim the stamp to avoid padding area
                this_object =  this_object[self.ystart:self.yend+1,self.xstart:self.xend+1]
                
                yss,xss = np.nonzero(this_object>0)
                
                if len(xss)<1:
                    continue 

                minx = np.min(xss)
                maxx = np.max(xss)
                miny = np.min(yss)
                maxy = np.max(yss)

                print("======>",minx,maxx,miny,maxy)
                this_object = this_object[miny:maxy+1,minx:maxx+1]

                dset = fhdf5.create_dataset("%d_%s" % (self.IDs[i],self.order),data=this_object,dtype='f',compression="gzip",compression_opts=9)
                dset.attrs[u'minx'] = minx
                dset.attrs[u'maxx'] = maxx
                dset.attrs[u'miny'] = miny
                dset.attrs[u'maxy'] = maxy
                dset.attrs[u'units'] = 'e-/s'


        if self.SBE_save != None:
            fhdf5.close()



    def disperse_chunk(self,c):
        """Method that handles the dispersion. To be called after create_pixel_list()"""
        from multiprocessing import Pool
        #from progressbar import Bar, ETA, ReverseBar, ProgressBar, Percentage
        import time

        if self.SED_file!=None:
            import h5py
            h5f = h5py.File(self.SED_file,'r')
            # b = h5f['16524'][:]
            pars = []
            ID = int(self.seg[self.ys[c][0],self.xs[c][0]])
            #print(ID)
            tmp = h5f["%s" % (ID)][:]
            for i in range(len(self.xs[c])):
                #ID = int(self.seg[self.ys[c][i],self.xs[c][i]])
                
                lams = tmp[0]
                fffs = tmp[1]*self.fs["SED"][c][i]
                #print("should be <<1 ",self.fs["SED"][c][i],tmp[1])
                f = [lams,fffs]
                xs0 = [self.xs[c][i],self.xs[c][i]+1,self.xs[c][i]+1,self.xs[c][i]]
                ys0 = [self.ys[c][i],self.ys[c][i],self.ys[c][i]+1,self.ys[c][i]+1]
                pars.append([xs0,ys0,f,self.order,self.C,ID,self.extrapolate_SED])
            h5f.close()

        else:
        # good code below
            pars = []
            for i in range(len(self.xs[c])):
                ID = i
                xs0 = [self.xs[c][i],self.xs[c][i]+1,self.xs[c][i]+1,self.xs[c][i]]
                ys0 = [self.ys[c][i],self.ys[c][i],self.ys[c][i]+1,self.ys[c][i]+1]
                lams = list(self.fs.keys())
                f = [lams,[self.fs[l][c][i] for l in self.fs.keys()]]
                pars.append([xs0,ys0,f,self.order,self.C,ID,self.extrapolate_SED])

        print(len(pars),"pixels loaded for dispersion...")
        
        time1 = time.time()
        mypool = Pool(self.max_cpu) # Create pool
        all_res = mypool.imap_unordered(helper,pars) # Stuff the pool
        mypool.close() # No more work

        #widgets=[Percentage(), Bar(), ETA()]
        #pbar = ProgressBar(widgets=widgets, maxval=len(pars)).start()

        #simulated_image = np.zeros(self.dims,np.float)
        this_object = np.zeros(self.dims,np.float)

        for i,pp in enumerate(all_res, 1):        
            if np.shape(pp.transpose())==(1,6):
                continue
            #print np.min(pp[0]),np.max(pp[0])
            x,y,w,f = pp[0],pp[1],pp[3],pp[4]

            vg = (x>=0) & (x<self.dims[1]) & (y>=0) & (y<self.dims[0]) 

            x = x[vg]
            y = y[vg]
            f = f[vg]
            w = w[vg]
            
            if len(x)<1:
                continue

            minx = int(min(x))
            maxx = int(max(x))
            miny = int(min(y))
            maxy = int(max(y))

            a = sparse.coo_matrix((f, (y-miny, x-minx)), shape=(maxy-miny+1,maxx-minx+1)).toarray()
            self.simulated_image[miny:maxy+1,minx:maxx+1] = self.simulated_image[miny:maxy+1,minx:maxx+1] + a
            this_object[miny:maxy+1,minx:maxx+1] = this_object[miny:maxy+1,minx:maxx+1] + a






        time2 = time.time()

        print(time2-time1,"s.")
        return this_object

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

