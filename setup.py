import os
from setuptools import setup, Extension, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name='NIRCAM_Gsim',
	version='1.4b1',
	description='NIRCAM Grism simulator, includes J.D. Smith polyclip C code',
    url='https://github.com/npirzkal/GRISM_NIRCAM',
	author='Nor Pirzkal',
	author_email='npirzkal@mac.com',
    package_dir = {
        'NIRCAM_Gsim': 'NIRCAM_Gsim',
        'NIRCAM_Gsim.polyclip': 'NIRCAM_Gsim/polyclip',
        'NIRCAM_Gsim.observations': 'NIRCAM_Gsim/observations'},
    packages=["NIRCAM_Gsim","NIRCAM_Gsim.polyclip","NIRCAM_Gsim.observations","NIRCAM_Gsim.disperse"],
    ext_modules=[Extension('NIRCAM_Gsim.polyclip.polyclip_c', ['NIRCAM_Gsim/polyclip/polyclip_c.c'])],
    install_requires=[
    "tqdm > 4.0.0",
    "grismconf >= 1.24"
],
)
