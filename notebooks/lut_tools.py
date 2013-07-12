#!/usr/bin/env python
"""
A set of utilities to create and plot RSE paper experiments

"""
import datetime
import sys

import numpy as np
# easy_install pyephem
import ephem   

from prosail import run_prosail


def xify(p,params = ['n','xcab','xcar','cbrown','xcw','xcm','xala','bsoil','psoil','hspot','xlai']):
  '''
  Fix the state key names by putting x on where appropriate
  '''
  retval = {}
  for i in p.keys():
    if 'x' + i in params:
      retval['x' + i] = p[i]
    else:
      retval[i] = p[i]
  return retval

def invtransform(params,logvals = {'ala': 90., 'lai':-2.0, 'cab':-100., 'car':-100., 'cw':-1/50., 'cm':-1./100}):
  retval = {}
  for i in xify(params).keys():
    if i[0] != 'x':
      retval[i] = params[i]
    else:
      rest = i[1:]
      try:
        if logvals[rest] < 0:
          retval[rest] = logvals[rest] * np.log(params[i])
        else:
          retval[rest] = params[i] * logvals[rest]
      except:
        retval[rest] = params[i]
  return retval

def transform(params,logvals = {'ala': 90., 'lai':-2.0, 'cab':-100., 'car':-100., 'cw':-1/50., 'cm':-1./100}):
  retval = {}
  for i in xify(params).keys():
    if i[0] != 'x':
      retval[i] = params[i]
    else:
      rest = i[1:]
      try:
        if logvals[rest] < 0:
          retval[i] = np.exp(params[rest] / logvals[rest])
        else:
          retval[i] = params[rest] / logvals[rest]
      except:
        retval[i] = params[i]
  return retval

def limits():
  '''
  Set limits for parameters
  '''
  params = ['n','xcab','xcar','cbrown','xcw','xcm','xala','bsoil','psoil','hspot','xlai']
  pmin = {}
  pmax = {}
  # set up defaults
  for i in params:
    pmin[i] = 0.001
    pmax[i] = 1. - 0.001
  # now specific limits
  # These from Feret et al.
  pmin['xlai'] = transform({'lai':15.})['xlai']
  pmin['xcab'] = transform({'cab':0.2})['xcab']
  pmax['xcab'] = transform({'cab':76.8})['xcab']
  pmin['xcar'] = transform({'car':25.3})['xcar']
  pmax['xcar'] = transform({'car':0.})['xcar']
  pmin['cbrown'] = 1.
  pmax['cbrown'] = 0.
  pmin['xcm'] = transform({'cm':0.0017})['xcm']
  pmax['xcm'] = transform({'cm':0.0331})['xcm']
  pmin['xcw'] = transform({'cw':0.0043})['xcw']
  pmax['xcw'] = transform({'cw':0.0713})['xcw']

  pmin['n'] = 0.8
  pmax['n'] = 2.5
  pmin['bsoil'] = 0.0
  pmax['bsoil'] = 2.0
  pmin['psoil'] = 0.0
  pmax['psoil'] = 1.0
  pmin['xala'] = transform({'ala':90.})['xala']
  pmax['xala'] = transform({'ala':0.})['xala']
  for i in params:
    if pmin[i] > pmax[i]:
      tmp = pmin[i]
      pmin[i] = pmax[i]
      pmax[i] = tmp
  return (pmin,pmax)


def fixnan(x):
  '''
  the RT model sometimes fails so we interpolate over nan

  This method replaces the nans in the vector x by their interpolated values
  '''
  for i in xrange(x.shape[0]):
    sample = x[i]
    ww = np.where(np.isnan(sample))
    nww = np.where(~np.isnan(sample))
    sample[ww] = np.interp(ww[0],nww[0],sample[nww])
    x[i] = sample
  return x

def addDict(input,new):
  '''
  add dictionary items
  '''
  for i in new.keys():
    try:
      input[i] = np.hstack((np.atleast_1d(input[i]),np.atleast_1d(new[i])))
    except:
      input[i] = np.array([new[i]]).flatten()
  return input

def unpack(params):
  '''
  Input a dictionary and output keys and array
  '''
  inputs = []
  keys = np.sort(params.keys())
  for i,k in enumerate(keys):
    inputs.append(params[k])
  inputs=np.array(inputs).T
  return inputs,keys

def pack(inputs,keys):
  '''
  Input keys and array and output dict
  '''
  params = {}
  for i,k in enumerate(keys):
    params[k] = inputs[i]
  return params

def sampler(pmin,pmax):
  '''
  Random sample
  '''
  params = {}
  for i in pmin.keys():
    params.update(invtransform({i:pmin[i] + np.random.rand()*(pmax[i]-pmin[i])}))
  return params


def samples(n=2):
  '''
  Random samples over parameter space
  '''
  (pmin,pmax) = limits()
  s = {}
  for i in xrange(n):
    s = addDict(s,sampler(pmin,pmax))
  return s




def brdfModel(s,vza,sza,raa):
  '''
  Run the full BRDF model for parameters in s
  '''
 
  try:
    brdf = []
    for i in xrange(len(s['n'])):
      brdf.append(run_prosail(s['n'][i],s['cab'][i],s['car'][i],s['cbrown'][i],s['cw'][i],s['cm'][i],\
                s['lai'][i],s['ala'][i],0.0,s['bsoil'][i],s['psoil'][i],s['hspot'][i],\
                vza[i],sza[i],raa[i],2))
  except:
    brdf = []
    for i in xrange(len(s['n'])):
      brdf.append(run_prosail(s['n'][i],s['cab'][i],s['car'][i],s['cbrown'][i],s['cw'][i],s['cm'][i],\
                s['lai'][i],s['ala'][i],0.0,s['bsoil'][i],s['psoil'][i],s['hspot'][i],\
                vza,sza,raa,2))
  return (vza,sza,raa),transform(s),fixnan(np.array(brdf))


def lut(n=2,vza=0.,sza=45.,raa=0.,brdf=None,bands=None):
  '''
  Generate a random LUT for given viewing and illumination geometry
  '''
  # get the parameter limits
  (pmin,pmax) = limits()

  # generate some random samples over the parameter space
  s = samples(n=n)
  return brdfModel(s,vza,sza,raa)

def create_surface_functions ( output_fname, param_defaults=None ):
    """
    This function creates a set of annual time-varying parameters parameters 
    for the RSE paper experiments. The function receives a set of parameter 
    names, by default these are the ones that are used by PROSPECT+PRICE+0.5Disc

    """
    if param_defaults is None:
        param_names = [ "gamma_time", "xlai", "xhc", "rpl", "xkab", "xkar", "xkw", "xkm", \
            "xleafn", "xs1", "xs2", "xs3", "xs4", "lad" ]
        param_defaults = { "xlai":0.995, "xhc":1, "rpl":0.01, "xkab":0.670, \
            "xkar":1.0, "xkw":0.06065, "xkm":0.3679, \
            "xleafn":1.0, "xs1":0.2, "xs2":0.0, "xs3":0.0, "xs4":0.0, \
            "lad":5, "gamma_time":1 }
    n_params = len ( param_defaults.keys() )
    doys = np.arange ( 1, 366    )
    # This is the time axis, daily for one year
    # the output array. It is a daily list of parameters, plus a time axis
    # and mask fields
    data = np.zeros ( ( 365, n_params +2 ) )
    data[:,0] = doys
    data[:,1] = 1 # Mask set to true for all days
    datastr = 'time mask' # This will in turn be the label in the param file
    for ( n, param_name ) in enumerate ( param_names ):
        def_value = param_defaults [ param_name ]
        t = (doys-1)/365. # Normalised DoY
        datastr = datastr + ' %s' % param_name
        # This loop decides whether we calculate a temporal trajectory for the
        # parameter
        if param_name == 'xlai':
            data[:,n+2] = 0.21 + 3.51 * (np.sin(np.pi*t)**5)
            # This is the parameter transormation
            data[:,n+2] = np.exp(-data[:,n+2]/2.)
        elif param_name == 'xkab':
            w = np.where(t<=0.5)[0]
            data[w,n+2] = 10.5 + 208.7*t[w]
            w = np.where(t>0.5)[0]
            data[w,n+2] = 219.2 - 208.7*t[w]
            # This is the parameter transormation
            data[:,n+2] = np.exp(-data[:,n+2]/100.)
        elif param_name == 'xkw':
            data[:,n+2] = 0.012 + 0.01*np.sin(np.pi * t+0.1) * np.sin(6*np.pi*t + 0.1)
            # This is the parameter transormation
            data[:,n+2] = np.exp(-data[:,n+2]*50.)
        elif param_name == 'xs1':
            data[:,n+2] = 0.2 + 0.18*np.sin(np.pi*t) * np.sin(6*np.pi*t)
        else:
            # No temporal trajectory for this parameter, just set it to its
            # default value
            data[:,n+2] = def_value


    # The dataset has now been created. We'll save it to a file, not forgetting
    # to add a header   
    fp = open ( output_fname, 'w' )
    fp.write ( '#PARAMETERS %s\n' % datastr )
    for d in np.arange(365):
        line_out = "".join ( [ "%12.6G " % i for i in data[d,:] ] )
        fp.write ( line_out + "\n" )
    fp.close()



class Sensor ( object ):
    """
    A class to forward model a set of parameters to a typical sensor 
    configuration in terms of orbital characteristics, wavebands, noise figures
    and other things.
    """
    def __init__ ( self, parameter_file ):
        """The constructor just reads a parameter file"""
        self.parameter_fname = parameter_file
        try:
            self.parameters = np.loadtxt( parameter_file )
            
        except ValueError:
            print "There was an error reading the parameter file %s" % \
                    parameter_file
            sys.exit( - 1)
            
    def set_wavebands ( self, wv, bw=None ):
        """This method allows setting the wavebands for this sensor. Bands
        are passed in `wv` as an array either of text, or if `bw` is set, then
        `wv` is interpreted as a numeric array with the cetnre wavelength, and
        `bw` give the bandwidth associated to each band. """
        self.n_bands = len ( wv )
        self.wbands = np.zeros ( (3, self.n_bands ) )
        if bw is None:
            for ( i, b ) in enumerate ( wv ):
                wb_min = float ( b.split("-")[0] ) 
                wb_max = float ( b.split("-")[1] ) 
                wb_mean = 0.5 * ( wb_min + wb_max )
                self.wbands [ 0, i ] = wb_mean
                self.wbands [ 1, i ] = wb_min
                self.wbands [ 2, i ] = wb_max
        else:
            
            for ( i, b ) in enumerate ( wv ):
                self.wbands [ 0, i ] = b
                self.wbands [ 1, i ] = b - 0.5*bw[i]
                self.wbands [ 2, i ] = b + 0.5*bw[i]
                
            
    def set_cloudiness ( self, prop=1.0, window=1., doy_list=None ):
        """
        This method simulates correlated observation loss by creating a mask.
        Alternatively, the user may provide a one year mask in `user_mask`.
        """
        if doy_list is None:
            weightings = np.repeat(1.0, window) / window
            
            xx = np.convolve( np.random.rand( len(self.doys)*100 ), \
                weightings, 'valid' )[window:window+len(self.doys)]
            
            maxx = sorted(xx)[:int(len(xx)*prop)]
            self.cloud_mask = np.in1d( xx, maxx )
        else:
            self.cloud_mask = doy_list
        return self.cloud_mask
            
    
    def set_acq_geometry ( self, lat, vza_spread, \
            lon="0:0", overpass_time="10:30" ):
        """
        This method simulates an acquisition geometry  for a nadir overpass and 
        a complete year.
        """
        
        self.doys = np.arange ( 1, 366 )
        
        self.vza = np.zeros ( (365) )
        self.vaa = np.zeros ( (365) )
        self.sza = np.zeros ( (365) )
        self.saa = np.zeros ( (365) )
        self.vaa = np.random.rand ( (365) )*360. # 
        self.saa = np.random.rand ( (365) )*360. # Random Azimuth angle
        self.vza = np.random.rand ( (365) )*vza_spread # Random wihtin bounds VZ
        observer = ephem.Observer()
        observer.lat, observer.long, observer.date = lat, lon, \
        datetime.datetime(2011, 1, 1, int(overpass_time.split(":")[0]), \
            int(overpass_time.split(":")[1]) )
        sun = ephem.Sun( observer )
        dd = observer.date
        for ( i,  doy ) in enumerate ( self.doys ):
            observer.date = dd + 1
            sun = ephem.Sun ( observer )
            dd = observer.date
            self.sza [i] = 90. - float(sun.alt)*180./np.pi                                
                                
    
    def set_noise ( self, b0, b1 ):
        """
        This method provides uncorrelated noise, following a linear spectral thingy
        """
        lambda_min = self.wbands[0,:].min()
        lambda_max = self.wbands[0,:].max()
        self.bu = np.zeros ( ( self.n_bands ) )
        self.bu = b0 + ((b1-b0)/(lambda_max-lambda_min)) * \
                (self.wbands[0,:] - lambda_min)
        
    def set_revisit ( self, revisit_time, shift=1, cloudy=False ):
        passer = self.doys % revisit_time == shift 
        
        if cloudy:
            #self.cloud_mask
            self.acq_dates = self.doys[ np.logical_and ( passer, \
                self.cloud_mask) ]
        else:
            self.acq_dates = self.doys[ passer ]

    def dump_empty_obs ( self, fname_out ):
        """
        This method creates an empty observations file. It is needed to run
        the model forward and fill it in with the appropriate simulated 
        reflectance. 
        """
        fp = open ( fname_out, 'w' )
        param_line = "#PARAMETERS time mask vza vaa sza saa "
        for b in xrange ( self.wbands.shape[1] ):
            param_line = param_line + "%g-%g " % \
                ( self.wbands[ 1, b], self.wbands[ 2, b] )
        for b in xrange ( self.wbands.shape[1] ):
                param_line = param_line + "sd-%g-%g " % \
                ( self.wbands[ 1, b], self.wbands[ 2, b] )
                
        fp.write ( param_line + "\n" )
        doys_list = np.in1d ( self.doys, self.acq_dates )
        for ( i, selecta ) in enumerate ( doys_list ):
            if selecta:
                out_line = "%d 1 " % self.doys[i]
                out_line = out_line + "%f %f %f %f " % ( self.vza[i], \
                    self.vaa[i], self.sza[i], self.saa[i] )
                for b in xrange ( self.wbands.shape[1] ):
                    out_line = out_line + "%12.6g " % 0 # Told u it was empty!
                    
                for b in xrange ( self.wbands.shape[1] ):
                    out_line = out_line + "%12.6g " % \
                    ( self.bu[b] )
                fp.write ( out_line + "\n" )
        fp.close()
                    
    
class Sentinel ( Sensor ):
    """
    This is a sentinel2/MSI type sensor. It is just a subclass of sensor to 
    set up a number of configuration options easily.
    """
    def __init__ ( self, parameter_file ):
        Sensor.__init__ ( self, parameter_file )
        self.set_wavebands(['433-453','457.5-522.5','542.5-577.5', \
            '650-680','697.5-712.5','732.5-747.5','773-793','784.5-899.5', \
            '855-875','935-955','1565-1655','2100-2280'] )
        self.set_noise( 0.008, 0.02 )
        self.set_acq_geometry ( 50., 15. )
        self.set_cloudiness( prop=1., window=1.)
        self.set_revisit ( 5, shift=0, cloudy=True)
        
    def eoldas_config ( self, fname ):
        
        fp = open ( fname, 'w' )
        eoldas_config_file="""
[operator.obs.rt_model]
model=rtmodel_ad_trans_Price
use_median=True
help_use_median = "Flag to state whether full bandpass function should be used or not."
bounds = [400,2500,1]
help_bounds = "The spectral bounds (min,max,step) for the operator'

[operator.obs.x]
names = $parameter.names[1:]
sd = [1.0]*len($operator.obs.x.names)
datatype = x

[operator.obs.y]
control = 'mask vza vaa sza saa'.split()
names = ['433-453','457.5-522.5','542.5-577.5','650-680','697.5-712.5','732.5-747.5','773-793','784.5-899.5','855-875','935-955','1565-1655','2100-2280']
sd = ["0.004", "0.00416142",  "0.00440183", "0.00476245", "0.00489983", "0.00502003","0.00516772", "0.00537035", "0.00544934", "0.0057241", "0.00800801","0.01" ]
datatype = y
state = 'input/rse1/rse1_test.GAMMA.dat'
help_state='set the obs state file'

[operator.obs.y.result]
filename = 'output/rse1/rse1_test_fwd.GAMMA.dat'
help_filename = 'forward modelling results file'
format = 'PARAMETERS'
        """
        fp.write ( eoldas_config_file )
        fp.close()
        self.this_conf = fname
        
        
    def run_model_fwd ( self, obs_file, output_file ):
        """
        Runs the  EOLDAS system on an empty observations file, and fills it with
        reflectance value, as well as adding noise to it.
        """
        xlai = 0.995
        xhc = 1
        rpl = 0.01
        xkab = 0.670
        xkar = 1.0
        xkw = 0.6065
        xkm = 0.3679
        xleafn = 1
        xs1 = 0.2
        xs2 = 0
        xs3 = 0
        xs4 = 0
        lad = 5
        gamma = 100
        defaults = '--parameter.x.default=%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d'%(gamma,xlai,xhc,rpl,xkab,xkar,xkw,xkm,xleafn,xs1,xs2,xs3,xs4,lad)
        #../EOLDAS_clean_noise2/eoldaslib/eoldas.py -c rse_expts.conf -c sentinel.conf --parameter.x.default=100.000000,0.995000,1.000000,0.010000,0.670000,1.000000,0.606500,0.367900,1.000000,0.200000,0.000000,0.000000,0.000000,5 --operator.obs.y.state=/tmp/sentinel_empty.dat --parameter.x.state=/tmp/test_me.dat --no_calc_posterior_unc --operator.modelt.rt_model.model_order=1 --operator.obs.y.result.filename=obsyresult.dat --parameter.result.filename=tmp/parameter.runtime --operator.prior.y.result.filename=tmp/prior.runtime --passer
        
        cmd_line =[ "../EOLDAS_clean_noise2/eoldaslib/eoldas", \
                "--conf=eoldas_config_sentinel.conf", \
                "--conf=rse_expts.conf", "--conf=%s"% ( self.this_conf ), \
                "%s"% ( defaults), \
                "--operator.obs.y.state=%s" % obs_file, \
                "--parameter.x.state=%s" % self.parameter_fname, \
                "--no_calc_posterior_unc", \
                "--operator.modelt.rt_model.model_order=1", \
                "--operator.obs.y.result.filename=%s" % output_file, \
                "--parameter.result.filename=tmp/parameter.runtime", \
                "--operator.prior.y.result.filename=tmp/prior.runtime", \
                "--passer" ]
                
                
                
                
                
                
        print cmd_line
        EOLDAS = eoldas ( cmd_line )
        EOLDAS.solve()
            
class SpotHRV ( Sensor ):
    """
    This is a sentinel2/MSI type sensor. It is just a subclass of sensor to 
    set up a number of configuration options easily.
    """
    def __init__ ( self, parameter_file ):
        Sensor.__init__ ( self, parameter_file )
        self.set_wavebands( [ '500-590', '610-680', '790-890', '1530-1750'])
        self.set_noise( 0.008, 0.02 )
        self.set_acq_geometry ( 50., 15. )
        self.set_cloudiness( prop=1., window=1.)
        self.set_revisit ( 13, shift=4, cloudy=True)

    def eoldas_config ( self, fname ):
        eoldas_config_file = """
        [operator.spot_obs.rt_model]
model=rtmodel_ad_trans_Price1
use_median=True
help_use_median = "Flag to state whether full bandpass function should be used or not."
bounds = [400,2500,1]
help_bounds = "The spectral bounds (min,max,step) for the operator'

[operator.spot_obs.x]
names = $parameter.names[1:]
sd = [1.0]*len($operator.obs.x.names)
datatype = x

[operator.spot_obs.y]
control = 'mask vza vaa sza saa'.split()
names = [ '500-590', '610-680', '790-890', '1530-1750']
sd = ['0.004', '0.00455','0.562','0.01']
datatype = y
state = 'input/rse1/hyper_test.GAMMA.dat'
help_state='set the obs state file'

[operator.spot_obs.y.result]
filename = 'output/rse1/hyper_test_fwd.GAMMA.dat'
help_filename = 'forward modelling results file'
format = 'PARAMETERS'
        """
        fp = open ( fname, 'w' )
        fp.write ( eoldas_config_file )
        fp.close()
        self.this_conf = fname
        
    def run_model_fwd ( self, obs_file, output_file ):
        """
        Runs the  EOLDAS system on an empty observations file, and fills it with
        reflectance value, as well as adding noise to it.
        """
        xlai = 0.995
        xhc = 1
        rpl = 0.01
        xkab = 0.670
        xkar = 1.0
        xkw = 0.6065
        xkm = 0.3679
        xleafn = 1
        xs1 = 0.2
        xs2 = 0
        xs3 = 0
        xs4 = 0
        lad = 5
        gamma = 100
        defaults = '--parameter.x.default=%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d'%(gamma,xlai,xhc,rpl,xkab,xkar,xkw,xkm,xleafn,xs1,xs2,xs3,xs4,lad)
        cmd_line =[ "../EOLDAS_clean_noise2/eoldaslib/eoldas", \
            "--conf=eoldas_config.conf", \
            "--conf=rse_expts_spot.conf", "--conf=%s"% ( self.this_conf ), \
            "%s"% ( defaults), \
            "--operator.spot_obs.y.state=%s" % obs_file, \
            "--parameter.x.state=%s" % self.parameter_fname, \
            "--no_calc_posterior_unc", \
            "--operator.modelt.rt_model.model_order=1", \
            "--operator.spot_obs.y.result.filename=%s" % output_file, \
            "--parameter.result.filename=tmp/parameter.runtime", \
            "--operator.prior.y.result.filename=tmp/prior.runtime", \
            "--passer" ]
        
        print cmd_line
        EOLDAS = eoldas ( cmd_line )
        EOLDAS.solve()



    
if __name__ == "__main__":
    
    create_surface_functions ("/tmp/test_me.dat" )
    sentinel = Sentinel ( "/tmp/test_me.dat" )
    spot = SpotHRV ( "/tmp/test_me.dat" )
    annual_cloud_mask = sentinel.set_cloudiness ( prop=0.5, window=10 )
    sentinel.set_revisit ( 5, shift=0, cloudy=True)
    spot.set_cloudiness ( prop=0.5, window=10, doy_list=annual_cloud_mask )
    spot.set_revisit ( 13, shift=4, cloudy=True )
    spot.dump_empty_obs ( "/tmp/spot_empty.dat")
    sentinel.dump_empty_obs ( "/tmp/sentinel_empty.dat")
    spot.eoldas_config ( "spot.conf" )
    sentinel.eoldas_config ("sentinel.conf")
    #sentinel.parameter_fname="truth.400.dat"
    
    sentinel.run_model_fwd ( "/tmp/sentinel_empty.dat", "sentinel_out.dat" )
    spot.run_model_fwd ( "/tmp/spot_empty.dat", "spot_out.dat" )