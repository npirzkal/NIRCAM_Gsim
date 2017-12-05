#! /usr/bin env python

"""
Code to generate NIRCAM dispersed seed images. Starting with a set of imaging seed images, 
potentially with padding, given a NIRCAM GRISMCONF configuration file
"""

import os
from astropy.io import fits
import numpy as np
from .observations.observations \
        import observation as NIRCAM_Gsim_observation 

class Grism_seed():
	def __init__(self,image_seeds,cross_filter,mode,config_path="."):
		# image_seeds: "V4*.fits"
		# mode: "modA_C" for module A and grismC
		# cross_filter: "F444W"
		# config_path: "/Users/GRISMDATA/NIRCAM/"
		config = os.path.join(config_path,"NIRCAM_%s_%s.conf" % (cross_filter,mode))
		self.config = config
		
		self.image_seeds = image_seeds
		self.cross_filter = cross_filter
		self.mode = mode
		self.config_path = config_path

		# Get information about input padding. We use the first image seed for this, just like for the segmentation info.
		h = fits.open(image_seeds[0])[0].header
		self.xstart = h["NOMXSTRT"]
		self.xend = h["NOMXEND"]
		self.ystart = h["NOMYSTRT"]
		self.yend = h["NOMYEND"]

		#print(self.xstart,self.xend,self.xend-self.xstart)

		# Get segmenationation info, from the first image seed.
		self.seg_data = fits.open(image_seeds[0])[2].data

		# TEST
		self.seg_data[self.seg_data!=1926.]=0.

	def observation(self,orders=["+1","+2"],max_split=100):
		# order: dispersion order, e.g. "+1"
		# max_split: we use max_split groups of pixels to disperse the whole image
		self.this_one = {}
		for order in orders:
			self.this_one[order] = NIRCAM_Gsim_observation(self.image_seeds,self.seg_data,self.config,order=order,max_split=max_split)
			self.this_one[order].disperse_all()

	def finalize(self,tofits=None,noBack=False):
		if noBack==False:
			final = fits.open(os.path.join(self.config_path,"%s_%s_back_V2.1.fits" % (self.cross_filter,self.mode)))[0].data
		else:
			final = 0.
		for order in self.this_one.keys():
			print(order)
			sim = self.this_one[order].simulated_image[self.ystart:self.yend+1,self.xstart:self.xend+1]
			final = final + sim
		self.final = final	
		print(np.shape(final))
		if tofits!=None:
			self.saveSingleFits(tofits)

	def saveSingleFits(self,name):
		#save an array into the first extension of a fits file
		h0 = fits.PrimaryHDU()
		h1 = fits.ImageHDU(self.final,name='DATA')
        
        
		hdulist = fits.HDUList([h0,h1])
		hdulist.writeto(name,overwrite=True)

if __name__ == '__main__':
	import glob

	image_seeds = glob.glob("V4*.fits")
	seed = Grism_seed(image_seeds,"F444W","modA_R","/Users/npirzkal/Dropbox/GRISMDATA/NIRCAM/")



