==============
eoldas_ng
==============
:Info: EO-LDAS (new/next) generation: a new take on the EO-LDAS codebase
:Author: J Gomez-Dans <j.gomez-dans@ucl.ac.uk>
:Date: 2015-12-11 16:00:00 +0000 
:Description: README file

.. image:: http://www.nceo.ac.uk/images/NCEO_logo_lrg.jpg
   :scale: 50 %
   :alt: NCEO logo
   :align: right
   
.. image:: http://www.esa.int/esalogo/images/logotype/img_colorlogo_darkblue.gif
   :scale: 50 %
   :alt: ESA logo
   :align: left
   

What's eoldas_ng?
--------------------

The ``eoldas_ng`` package provides a flexible variational data assimilation framework for EO image interpretation. The idea behind this package stem from the observation that the inversion of physical models to estimate land surface parameters is *ill posed*. It is therefore important to add prior constraints, in the form of prior parameter distributions, or models (mechanistic or empirical) that provide us with an estimate of the parameter prior to the assimilation of the observations. 

``eoldas_ng`` tries to help users that are interested in inferring the state of the land surface using EO data and physical models by providing an easy to use, flexible library, written in Python that allows them to deploy fairly sophisticated retrieval procedures.

The current library can be used to consistently retrieve land surface parameters such as LAI, leaf chlorophyll concentration, etc. from different sensors, with different spectral, spatial and angular acquisition characteristics. 

The library is **PRE-RELEASE** although some users will find it useful ;-) We are developing the library, a work that is being funded by 

* NCEO in the UK
* ESA, through various projets
