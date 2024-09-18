#! /usr/bin env python

"""
Code to generate WFSS dispersed seed images. Starting with a set of imaging seed images, 
potentially with padding, given a JWST WFSS GRISMCONF configuration file
"""

import os, pickle
from astropy.io import fits
import numpy as np
from .observations.observations \
import observation as Gsim_observation 
import multiprocessing

class Grism_seed():
    def __init__(self,image_seeds,cross_filter,mode,config_path=".",extrapolate_SED=False,SED_file=None,instrument="NIRCAM",max_cpu=None, SBE_save=None, renormalize=True, resample=False, outbound=False, save_objects=None):
        """A class for a grism simulation

        Attributes
        ----------
        image_seeds: list
            A list of image image_seeds or of seed cube files
        cross_filter: str
            A string containing the name of a direct filter
        mode: str
            A string containing either R or C
        config_path: str
            As string pointing to a GRISMCONF configuration file
        extrapolate_SED: bol
            If set to True, the objects' SED will be extrapolated if needed
        SED_file: str
            A string containing the name of an HDF5 file containing the spectra of each source
        instrument: str
            A string containing the name of the instrument (e.g. NIRCAM)
        max_cpu: int
            An integer containing the number of CPU to use in the multiheaded pool when dispersing
        SBE_save: str
            A string containing the name of an output HDF5 file which will contain simulated 2D stamps of each source
        renormalize: bol
            Whether to renormalize the input data to unity over segmentation map area when using an input spectrum.
        reaample: bol
            If true, the disperser will first smooth and resample input spectra appropriately to match the disperser resolution.
        save_objest: None or filename
            Save computed objects footprints for future use

        Methods
        -------
        observation(self,orders=["+1","+2"],max_split=-1000,ID=0)

        finalize(self,tofits=None,Back=False,BackLevel=None)

        saveSingleFits(self,name)

        """

        config = os.path.join(config_path,"%s_%s_%s.conf" % (instrument,cross_filter,mode))
        self.config = config
        
        self.image_seeds = image_seeds
        self.c3d = None
        self.cross_filter = cross_filter
        self.mode = mode
        self.config_path = config_path
        if max_cpu is None:
            max_cpu = multiprocessing.cpu_count() - 1

        self.max_cpu = max_cpu
        self.SBE_save = SBE_save

        if self.SBE_save != None:
            print("Will output to ", self.SBE_save)
            if os.path.isfile(self.SBE_save):
                os.unlink(self.SBE_save)


        # Get information about input padding. We use the first image seed for this, just like for the segmentation info.
        h = fits.open(image_seeds[0])[0].header
        self.xstart = int(h["NOMXSTRT"])
        self.xend = int(h["NOMXEND"])
        self.ystart = int(h["NOMYSTRT"])
        self.yend = int(h["NOMYEND"])

        # Get segmentation info, from the first image seed.
        self.seg_data = fits.open(image_seeds[0])[2].data

        self.extrapolate_SED = extrapolate_SED
        self.SED_file = SED_file
        self.renormalize = renormalize
        self.resample = resample
        self.outbound = outbound
        self.save_objects = save_objects

        print("Initializing Pool with",self.max_cpu,"cores")
        self.mypool =  multiprocessing.Pool(processes=self.max_cpu)

        # temp 
        # Check if we have seed cube and use these instead... we just load them 
        print("image_seeds:",image_seeds)

        if "_ptsrc_seed_image.fits" in image_seeds[0]:
            sc3d = image_seeds[0].split("_uncal")[0]+"_uncal_star_seed_cube.pickle"
            if os.path.isfile(sc3d):
                print("ptsrc seg seed data found")
                self.c3d = pickle.load(open(sc3d,"rb"))
        if "_extended_seed_image.fits" in image_seeds[0]:
            sc3d = image_seeds[0].split("_uncal.fits")[0]+"_uncal_galaxy_seed_cube.pickle"
            if os.path.isfile(sc3d):
                print("extended seg seed data found")
                self.c3d = pickle.load(open(sc3d,"rb"))

    def observation(self,orders=None,max_split=-1000,ID=0):
        """Sets up an observation.

        Parameters
        ----------
        orders : list
            A list of string containing the name of the orders to disperse
        max_split : int
            Maximum number of pixels to disperse at once (not currently used)
        ID : int
            Specific object ID to dispersed. Set to 0 (default) to disperse all of available objects
        """

        self.this_one = {}

        # If orders are not passed, we get them from the config file
        
        if orders==None:
            import grismconf
            C = grismconf.Config(self.config)
            print("orders:",C.orders)
            self.orders = C.orders
        else:
            self.orders = orders

        for order in self.orders:
            boundaries = [self.xstart,self.xend,self.ystart,self.yend]
            self.this_one[order] = Gsim_observation(self.mypool,self.image_seeds,self.seg_data,self.config,seed_cube=self.c3d,order=order,max_split=max_split,extrapolate_SED=self.extrapolate_SED,SED_file=self.SED_file,max_cpu=self.max_cpu,ID=ID, SBE_save=self.SBE_save,boundaries=boundaries,renormalize=self.renormalize,resample=self.resample, outbound=self.outbound, save_objects=self.save_objects)
            #self.this_one[order].disperse_all()

    def disperse(self,orders=None,cache=False,trans=None):
        """Run the disperser. 

        Parameters
        ----------
        orders: list
            Optional list containing the name of the orders to disperse
        cache: bool
            If set to True, the dispersion tables are cached and will be used on subsequent calls. 
        """
        if orders==None:
            orders = self.orders

        print("Dispersing orders ", orders)
        for order in orders:
            #print("Dispersing order ",order)
            if self.this_one[order].cache:
                self.this_one[order].disperse_all_from_cache(trans=trans)
            else:
                self.this_one[order].disperse_all(cache=cache)

    def disperse_background_1D(self,background):
        """Produces a dispersed 2D image of the background spectrum contained in the fits file background, meant to be the 
        output of thr jwst_background module. This background is dispersed either in the row or column direction, depending on the 
        dispersion, and the result is tiled to produce a full 2D image. All orders are generated and added up.

        Parameters
        ----------
        background: 2D numpy array [lambda values,flux values]
            A 2D numpy array containing the spectrum of the background to disperse. 
            The wavelength should be in (micron) and the flux (in Mjy/sr), as is produced by 
            the jwst_background package
        output: numpy 2D array
            A 2D array containing the model background which can be fed back into finalize()
        """

        bck = 0.
        for order in self.orders:
            print("Computing dispersed background for order ",order)
            bck += self.this_one[order].disperse_background_1D(background) 

#        fits.writeto("WFSS_background.fits",bck,overwrite=True)
        return bck

    def finalize(self,Back=None,BackLevel=None,tofits=None):
        """ Produces a 2D dispersed image and add the appropriate background

        Parameters
        ----------
        tofits: str 
            Name of a fits file to write the simulation to. Default is set to None
        Back: str
            Name of a fits file containing an image of the background in e-/s in extension 0
            If None, the file listed in the config file is used.
        BackLevel: float
            Renormalization factor for the background image. Renormalization is done by multiplying by BackLevel/np.median(Back)
        """

        # Initialize final image with the background estimate
        final = 0.
        if (Back is None) and (BackLevel is not None):
            # Use pre-computed background from config file, scaled by BackLevel
            import grismconf
            bck_file = grismconf.Config(self.config).BCK
            print("Adding pre-computed 2D dispersed background ",bck_file,"scaled to",BackLevel)
            import grismconf
            final = fits.open(bck_file)[1].data * BackLevel
        if (Back is None) and (BackLevel is None):
            # Use no background
            print("No background added")
            final = 0.
        if (type(Back)==np.ndarray) and (BackLevel is not None):
            # Use a passed background and scale its median to BackLevel
            print("adding passed background array scaled to",BackLevel)
            final = Back/np.median(Back) * BackLevel
        if (type(Back)==np.ndarray) and (BackLevel is None):
            # Use a passed background as is
            print("adding passed background array as is")
            final = Back

        if not ((Back is None) and (BackLevel is None)) and (tofits!=None):
            # Save the background image to a fits file
            hprime = fits.PrimaryHDU()
            himg = fits.ImageHDU(final)
            himg.header['EXTNAME'] = 'BACKGRND'
            himg.header['UNITS'] = 'e/s'
            if BackLevel is not None:
                himg.header['BackLevel'] = BackLevel
            hlist = fits.HDUList([hprime, himg])
            hlist.writeto(tofits, overwrite=True)

        for order in self.orders:
            print("Adding contribution from order ",order)
            try:
                sim = self.this_one[order].simulated_image[self.ystart:self.yend+1,self.xstart:self.xend+1]
            except AttributeError:
                print("Contribution from order",order,"is missing. Skipping it.")
                continue
            final = final + sim
        self.final = final  
        
        if (Back is None) and (BackLevel is None) and (tofits!=None):
            # Save the background image to a fits file
            hprime = fits.PrimaryHDU()
            himg = fits.ImageHDU(final)
            himg.header['EXTNAME'] = 'BACKGRND'
            himg.header['UNITS'] = 'e/s'
            if BackLevel is not None:
                himg.header['BackLevel'] = BackLevel
            hlist = fits.HDUList([hprime, himg])
            hlist.writeto(tofits, overwrite=True)

    def saveSingleFits(self,name):
        """A helper function to write the 2D simulated imae into a fits file

        Parameters
        ----------
        fname: str
            The file name to use for the output
        """

        #save an array into the first extension of a fits file
        h0 = fits.PrimaryHDU()
        h1 = fits.ImageHDU(self.final,name='DATA')
        
        
        hdulist = fits.HDUList([h0,h1])
        hdulist.writeto(name,overwrite=True)

if __name__ == '__main__':
    import glob

    image_seeds = glob.glob("V4*.fits")
    seed = Grism_seed(image_seeds,"F444W","modA_R","/Users/npirzkal/Dropbox/GRISMDATA/NIRCAM/")



