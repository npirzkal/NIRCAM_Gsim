#! /usr/bin env python

"""
Code to generate WFSS dispersed seed images. Starting with a set of imaging seed images, 
potentially with padding, given a JWST WFSS GRISMCONF configuration file
"""

import os
from astropy.io import fits
import numpy as np
from .observations.observations \
        import observation as Gsim_observation 


class Grism_seed():
	def __init__(self,image_seeds,cross_filter,mode,config_path=".",extrapolate_SED=False,SED_file=None,instrument="NIRCAM",max_cpu=5, SBE_save=None):
		"""A class for a grism simulation

		Attributes
    	----------
    	image_seeds: list
    		A list of image image_seeds
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

		Methods
    	-------
    	observation(self,orders=["+1","+2"],max_split=-1000,ID=0)

    	finalize(self,tofits=None,Back=False,BackLevel=None)

    	saveSingleFits(self,name)

		"""

		config = os.path.join(config_path,"%s_%s_%s.conf" % (instrument,cross_filter,mode))
		self.config = config
		
		self.image_seeds = image_seeds
		self.cross_filter = cross_filter
		self.mode = mode
		self.config_path = config_path
		self.max_cpu = max_cpu
		self.SBE_save = SBE_save

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

		for order in self.orders:
			self.this_one[order] = Gsim_observation(self.image_seeds,self.seg_data,self.config,order=order,max_split=max_split,extrapolate_SED=self.extrapolate_SED,SED_file=self.SED_file,max_cpu=self.max_cpu,ID=ID, SBE_save=self.SBE_save,boundaries=[self.xstart,self.xend,self.ystart,self.yend])
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
			print("Dispersing order ",order)
			if self.this_one[order].cache:
				self.this_one[order].disperse_all_from_cache(trans=trans)
			else:
				self.this_one[order].disperse_all(cache=cache)


	def finalize(self,tofits=None,Back=None,BackLevel=None):
		""" Produces a 2D dispersed image and add the appropriate background

		Parameters
		----------
		tofits: str 
			Name of a fits file to write the simulation to. Default is set to None
		Back: str
			Name of a fits file containing an image of the background in e-/s in extension 0
		BackLevel: float
			Renormalization factor for the background image. Renormalization is done by multiplying by BackLevel/np.max(Back)
		"""

		if Back!=None:
			final = fits.open(os.path.join(self.config_path,Back))[0].data
			if BackLevel!=None:
				final = final/np.max(final)*BackLevel
		else:
			final = 0.
		for order in self.orders:
			print("Adding contribution from order ",order)
			try:
				sim = self.this_one[order].simulated_image[self.ystart:self.yend+1,self.xstart:self.xend+1]
			except AttributeError:
				print("Contribution from order",order,"is missing. Skipping it.")
				continue
			final = final + sim
		self.final = final	
		if tofits!=None:
			self.saveSingleFits(tofits)

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



