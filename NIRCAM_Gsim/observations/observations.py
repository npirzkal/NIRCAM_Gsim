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
import tqdm

def comprehension_flatten( aList ):
        return list(y for x in aList for y in x)

def helper(vars):
    x0s,y0s,f,order,C,ID,extrapolate_SED, xoffset, yoffset = vars # in this case ID is dummy number
    p = dispersed_pixel(x0s,y0s,f,order,C,ID,extrapolate_SED=extrapolate_SED,xoffset=xoffset,yoffset=yoffset)
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
        boundaries: a tuple containing the coordinates of the FOV within the larger seed image.
        """

        self.C = grismconf.Config(config)
            
        if plot:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.clf()
            x = np.arange(self.C.WMIN,self.C.WMAX,(self.C.WMAX,self.C.WMIN)/100.)
            plt.plot(x,self.C.SENS[order](x))

        self.ID = ID
        self.IDs = []
        self.dir_images = direct_images
        self.seg = segmentation_data
        self.dims = np.shape(self.seg)
        self.order = order
        self.SED_file = SED_file
        self.SBE_save = SBE_save
        self.max_cpu = max_cpu
        self.cache = False


        # if SBE_save!=None:
        #     if len(boundaries)!=4:
        #         print("WARMING: boundaries needs to be specified if using SBE_save")
        #         sys.exit()

        if len(boundaries)!=4:
            xpad = (np.shape(segmentation_data)[1]-self.C.NAXIS[0])//2
            ypad = (np.shape(segmentation_data)[0]-self.C.NAXIS[1])//2
            self.xstart = 0 + xpad
            self.xend = xpad + self.C.NAXIS[0]-1
            self.ystart = 0 + ypad
            self.yend = ypad + self.C.NAXIS[1]-1
            print("No boundaries passed. Assuming symmetrical padding of {} {} pixels and a final size of {} {} .".format(xpad,ypad,self.xend+1-self.xstart,self.yend+1-self.ystart))
        else:
            self.xstart,self.xend,self.ystart,self.yend = boundaries

        
        self.extrapolate_SED = extrapolate_SED # Allow for SED extrapolation
        if self.extrapolate_SED:
            print("Warning: SED Extrapolation turned on.")

        self.apply_POM()
        self.create_pixel_list()
        
        self.p_l = []
        self.p_a = []


    def apply_POM(self):
        """Account for the finite size of the POM and remove pixels in segmentation files which should not
        be dispersed"""
        x0 = int(self.xstart+self.C.XRANGE[self.C.orders[0]][0] + 0.5)
        x1 = int(self.xend+self.C.XRANGE[self.C.orders[0]][1] + 0.5)
        y0 = int(self.ystart+self.C.YRANGE[self.C.orders[0]][0] + 0.5)
        y1 = int(self.yend+self.C.YRANGE[self.C.orders[0]][1] + 0.5)

        if x0<0: x0 = 0
        if y0<0: y0 = 0
        ymax,xmax = np.shape(self.seg)
        if x0+1>xmax: x0 = xmax-1
        if y0+1>ymax: y0 = ymax-1

        from astropy.io import fits
        print("POM footprint applied: [{}:{},{}:{}]".format(x0,x1,y0,y1))
        print("Pixels outside of this region of the input data ([{},{}]) will not be dispersed.".format(xmax,ymax))
        self.seg[:,:x0+1] = 0
        self.seg[:,x1:] = 0
        self.seg[:y0+1,:] = 0
        self.seg[y1:,:] = 0

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

                if (len(xs)>0) & (len(ys)>0):
                    self.xs.append(xs)
                    self.ys.append(ys)
                    self.IDs = all_IDs
        else:
            vg = self.seg==self.ID
            ys,xs = np.nonzero(vg)            
           
            if (len(xs)>0) & (len(ys)>0):    
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
    
    def disperse_all(self,cache=False):

        if cache:
            print("Object caching ON")
            self.cache = True
            self.cached_object = {}

        self.simulated_image = np.zeros(self.dims,np.float)

        for i in tqdm.tqdm(range(len(self.IDs))):
            if self.cache:
                self.cached_object[i] = {}
                self.cached_object[i]['x'] = []
                self.cached_object[i]['y'] = []
                self.cached_object[i]['f'] = []
                self.cached_object[i]['w'] = []
                self.cached_object[i]['minx'] = []
                self.cached_object[i]['maxx'] = []
                self.cached_object[i]['miny'] = []
                self.cached_object[i]['maxy'] = []

            this_object = self.disperse_chunk(i)

            if self.SBE_save != None:
                # If SBE_save is enabled, we create an HDF5 file containing the stamp of this simulated object
                # order is in self.order
                # We just save the x,y,f,w arrays as well as info about minx,maxx,miny,maxy
    
                # We trim the stamp to avoid padding area
                this_SBE_object =  this_object[self.ystart:self.yend+1,self.xstart:self.xend+1]
                
                yss,xss = np.nonzero(this_SBE_object>0)
                
                if len(xss)<1:
                    continue 

                minx = np.min(xss)
                maxx = np.max(xss)
                miny = np.min(yss)
                maxy = np.max(yss)

                this_SBE_object = this_SBE_object[miny:maxy+1,minx:maxx+1]

                if os.path.isfile(self.SBE_save):
                    mode = "a"
                else:
                    mode = "w"

                with h5py.File(self.SBE_save,mode) as fhdf5:
                    dset = fhdf5.create_dataset("%d_%s" % (self.IDs[i],self.order),data=this_SBE_object,dtype='f',compression="gzip",compression_opts=9)
                    dset.attrs[u'minx'] = minx
                    dset.attrs[u'maxx'] = maxx
                    dset.attrs[u'miny'] = miny
                    dset.attrs[u'maxy'] = maxy
                    dset.attrs[u'units'] = 'e-/s'


    def disperse_background_1D(self,background):
        """Method to create a simple disperse background, obtained by dispersing a full row or column.
        We assume no field dependence in the cross dispersion direction and create a full 2D image by tiling a single dispersed row or column"""

        # Create a fake object, line in middle of detector
        naxis = self.dims
        C = self.C
        xpos,ypos = naxis[0]//2,naxis[1]//2

        # Find out if this an x-direction or y-direction dispersion
        dydx = np.array(C.DISPXY(self.order,1000,1000,1))-np.array(C.DISPXY(self.order,1000,1000,0))
        if np.abs(dydx[0])>np.abs(dydx[1]):
            print("disperse_background_1D: x-direction")
            direction = "x"
            xs = np.arange(self.C.XRANGE[self.order][0]+0,self.C.XRANGE[self.order][1]+naxis[0])
            ys = np.zeros(np.shape(xs))+ypos
        else:
            print("disperse_background_1D: y-direction")
            direction = "y"
            ys = np.arange(self.C.YRANGE[self.order][0]+0,self.C.YRANGE[self.order][1]+naxis[0])
            xs = np.zeros(np.shape(ys))+xpos

        print(xpos,ypos)
        
        
        lam = background[0]
        fnu = background[1]

        fnu = fnu/4.25e10 # MJy/arcsec^2
        fnu = fnu*1e6 # Jy/arcsec^2
        fnu = fnu * (0.065**2) # Jy/pixel

        fnu = fnu*1e-23
        c = 299792458.* 1e10 # A
        wa = lam*10000
        flam = fnu/(wa**2/c)

        f = [lam,flam]
                
        pars = []        
        for i in range(len(xs)):
            ID = 1
            xs0 = [xs[i],xs[i]+1,xs[i]+1,xs[i]]
            ys0 = [ys[i],ys[i],ys[i]+1,ys[i]+1]
            pars.append([xs0,ys0,f,self.order,C,ID,False,self.xstart,self.ystart])


        from multiprocessing import Pool
        import time
        time1 = time.time()
        mypool = Pool(self.max_cpu) # Create pool
        all_res = mypool.imap_unordered(helper,pars) # Stuff the pool
        mypool.close()

        bck = np.zeros(naxis,np.float)
        for i,pp in enumerate(all_res, 1): 
            if np.shape(pp.transpose())==(1,6):
                continue
            x,y,w,f = pp[0],pp[1],pp[3],pp[4]


            vg = (x>=0) & (x<naxis[0]) & (y>=0) & (y<naxis[1]) 

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
            bck[miny:maxy+1,minx:maxx+1] = bck[miny:maxy+1,minx:maxx+1] + a

        if direction=="x":
            bck = np.sum(bck,axis=0)
            bck = np.tile(bck,[naxis[1],1])
        else:
            bck = np.sum(bck,axis=1)
            bck = np.tile(bck,[naxis[0],1]).transpose()

        return bck

    def disperse_chunk(self,c):
        """Method that handles the dispersion. To be called after create_pixel_list()"""
        from multiprocessing import Pool
        import time

        if self.SED_file!=None:
            # We use an input spectrum file
            import h5py
            with h5py.File(self.SED_file,'r') as h5f:
                pars = []
                ID = int(self.seg[self.ys[c][0],self.xs[c][0]])

                tmp = h5f["%s" % (ID)][:]
                for i in range(len(self.xs[c])):
                    
                    lams = tmp[0]
                    fffs = tmp[1]*self.fs["SED"][c][i]

                    # trim input spectrum 
                    try:
                        ok = (lams>self.C.WRANGE["+1"][0]) & (lams<self.C.WRANGE["+1"][1])
                    except:
                        ok = (lams>self.C.WRANGE["A"][0]) & (lams<self.C.WRANGE["A"][1])
                    lams = lams[ok]
                    fffs = fffs[ok]

                    f = [lams,fffs]
                    xs0 = [self.xs[c][i],self.xs[c][i]+1,self.xs[c][i]+1,self.xs[c][i]]
                    ys0 = [self.ys[c][i],self.ys[c][i],self.ys[c][i]+1,self.ys[c][i]+1]
                    pars.append([xs0,ys0,f,self.order,self.C,ID,self.extrapolate_SED,self.xstart,self.ystart])
        else:
            # No spectrum passed
            pars = []
            for i in range(len(self.xs[c])):
                ID = i
                xs0 = [self.xs[c][i],self.xs[c][i]+1,self.xs[c][i]+1,self.xs[c][i]]
                ys0 = [self.ys[c][i],self.ys[c][i],self.ys[c][i]+1,self.ys[c][i]+1]
                lams = np.array(list(self.fs.keys()))
                flxs = np.array([self.fs[l][c][i] for l in self.fs.keys()])
                ok = flxs!=0 # We avoid any pixel containing pure 0's
                if len(flxs[ok])==0: continue
                flxs = flxs[ok]
                lams = lams[ok]
                ok = np.argsort(lams)
                flxs = flxs[ok]
                lams = lams[ok]
                f = [lams,flxs]
                pars.append([xs0,ys0,f,self.order,self.C,ID,self.extrapolate_SED,self.xstart,self.ystart])



        time1 = time.time()
        mypool = Pool(self.max_cpu) # Create pool
        all_res = mypool.imap_unordered(helper,pars) # Stuff the pool
        mypool.close() # No more work

        this_object = np.zeros(self.dims,np.float)

        for i,pp in enumerate(all_res, 1): 

            if np.shape(pp.transpose())==(1,6):
                continue

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

            if self.cache:
                self.cached_object[c]['x'].append(x)
                self.cached_object[c]['y'].append(y)
                self.cached_object[c]['f'].append(f)
                self.cached_object[c]['w'].append(w)
                self.cached_object[c]['minx'].append(minx)
                self.cached_object[c]['maxx'].append(maxx)
                self.cached_object[c]['miny'].append(miny)
                self.cached_object[c]['maxy'].append(maxy)



        time2 = time.time()

        return this_object

    def disperse_all_from_cache(self,trans=None):
        if not self.cache:
            print("No cached object stored.")
            return

        self.simulated_image = np.zeros(self.dims,np.float)

        for i in tqdm.tqdm(range(len(self.IDs)),desc="Dispersing from cache"):
            this_object = self.disperse_chunk_from_cache(i,trans=trans)


    def disperse_chunk_from_cache(self,c,trans=None):
        """Method that handles the dispersion. To be called after create_pixel_list()"""
        
        import time

        if not self.cache:
            print("No cached object stored.")
            return

        time1 = time.time()
        
        this_object = np.zeros(self.dims,np.float)

        if trans!=None:
                print("Applying a transmission function...")
        for i in range(len(self.cached_object[c]['x'])): 
            x = self.cached_object[c]['x'][i]
            y = self.cached_object[c]['y'][i]
            f = self.cached_object[c]['f'][i]*1.
            w = self.cached_object[c]['w'][i]

            if trans!=None:
                f *= trans(w)

            minx = self.cached_object[c]['minx'][i]
            maxx = self.cached_object[c]['maxx'][i]
            miny = self.cached_object[c]['miny'][i]
            maxy = self.cached_object[c]['maxy'][i]
   
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

