import fnmatch
import datetime
import glob
import os
import re
import xml.etree.ElementTree
from collections import OrderedDict, namedtuple

from osgeo import gdal
import numpy as np
import scipy.sparse as sp

from eoldas_ng import Prior, SpatialSmoother
from eoldas_ng import ObservationOperatorImageGP, StandardStatePROSAIL
import gp_emulator

FIXED = 1
CONSTANT = 2
VARIABLE = 3
   
def get_vaa ( lrx, lry, ulx, uly ):
    
    dx = ulx - lrx
    dy = uly - lry
    m = np.sqrt ( dx*dx + dy*dy )
    r = np.atan2 ( dx/m, dy/m )
    d = -180.*r/np.pi
    if d < 0:
        d = -d
    
def set_prior (state, prev_date=None ):
    mu_prior = OrderedDict (  )
    prior_inv_cov = OrderedDict ()
    prior_inv_cov['n'] = np.array([1])
    prior_inv_cov['cab'] = np.array([1])
    prior_inv_cov['car'] = np.array([1])
    prior_inv_cov['cbrown'] = np.array([1])
    prior_inv_cov['cw'] = np.array([0.7])
    prior_inv_cov['cm'] = np.array([0.7])
    prior_inv_cov['lai'] = np.array([1])
    prior_inv_cov['ala'] = np.array([1])
    prior_inv_cov['bsoil'] = np.array([3])
    prior_inv_cov['psoil'] = np.array([3])
        
    for param in state.parameter_min.iterkeys():
        if state.transformations.has_key ( param ):
            mu_prior[param] = state.transformations[param]( \
                    np.array([default_par[param]]) )
        else:
            mu_prior[param] = np.array([default_par[param]])
        prior_inv_cov[param] = 1./prior_inv_cov[param]**2

    n_elems = state_grid.size 
    n_pars = 0
    for k,v in state.state_config.iteritems():
        if v == 3: # VARIABLE for the time being...
            n_pars += 1
            
    prior_vect = np.zeros ( n_elems*n_pars  )# 
    inv_sig_prior = np.zeros ( n_elems*n_pars  )# 6 variable params + 2 const
    # Now, populate said vector in the right order
    # looping over state_config *should* preserve the order
    if prev_date is not None:
        f = cPickle.load ( open ( prev_date, 'r'))
        x = f['transformed_map'] # needs to be vectorised, it's a dict
        prior_vect = state.pack_from_dict ( x, do_transform=False )
        prior_inv_covariance = f['hessian']
        
    else:    
        i = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT: # Constant value for all times
                if prev_date is None:
                    prior_vect[i] = mu_prior[param]
                else:
                    prior_vect[i] = previous_posterior['transformed_map'][param]
                inv_sig_prior[i] = prior_inv_cov[param]
                i = i+1        
            elif typo == VARIABLE:
                # For this particular date, the relevant parameter is at location iloc
                if prev_date is None:
                    prior_vect[i:(i + n_elems)] =  \
                        np.ones(n_elems)*mu_prior[param]
                else:
                    prior_vect[i:(i + n_elems)] = \
                        previous_posterior['transformed_map'][param].flatten()
                inv_sig_prior[i:(i + n_elems)] = \
                        np.ones ( n_elems)*prior_inv_cov[param]
                i += n_elems
        prior_inv_covariance = sp.dia_matrix( ([inv_sig_prior], [0]),\
            shape=[n_elems*n_pars, n_elems*n_pars])


    prior = Prior ( prior_vect, prior_inv_covariance )
    return prior


def reproject_image_to_master ( master, slave, res=None ):
    """This function reprojects an image (``slave``) to
    match the extent, resolution and projection of another
    (``master``) using GDAL. The newly reprojected image
    is a GDAL VRT file for efficiency. A different spatial
    resolution can be chosen by specifyign the optional
    ``res`` parameter. The function returns the new file's
    name.
    
    Parameters
    -------------
    master: str 
        A filename (with full path if required) with the 
        master image (that that will be taken as a reference)
    slave: str 
        A filename (with path if needed) with the image
        that will be reprojected
    res: float, optional
        The desired output spatial resolution, if different 
        to the one in ``master``.
        
    Returns
    ----------
    The reprojected filename
    TODO Have a way of controlling output filename
    """
    slave_ds = gdal.Open( slave )
    if slave_ds is None:
        raise IOError, "GDAL could not open slave file %s " \
            % slave
    slave_proj = slave_ds.GetProjection()
    slave_geotrans = slave_ds.GetGeoTransform()
    data_type = slave_ds.GetRasterBand(1).DataType
    n_bands = slave_ds.RasterCount
    
    master_ds = gdal.Open( master )
    if master_ds is None:
        raise IOError, "GDAL could not open master file %s " \
            % master
    master_proj = master_ds.GetProjection()
    master_geotrans = master_ds.GetGeoTransform()
    w = master_ds.RasterXSize
    h = master_ds.RasterYSize
    if res is not None:
        master_geotrans[1] = float( res )
        master_geotrans[-1] = - float ( res )
    
    dst_filename= slave.replace( ".tif", "_crop.vrt" )
    dst_ds = gdal.GetDriverByName('VRT').Create(dst_filename, \
        w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( master_geotrans )
    dst_ds.SetProjection( master_proj)

    
    gdal.ReprojectImage( slave_ds, dst_ds, slave_proj, \
        master_proj, gdal.GRA_NearestNeighbour)
    dst = None # Flush to disk
    return dst_filename 

def reproject_cut ( slave, box=None, t_srs=None, s_srs=None, res=None ):
    """This function reprojects an image (``slave``) to
    match the extent, resolution and projection of another
    (``master``) using GDAL. The newly reprojected image
    is a GDAL VRT file for efficiency. A different spatial
    resolution can be chosen by specifyign the optional
    ``res`` parameter. The function returns the new file's
    name.
    
    Parameters
    -------------
    master: str 
        A filename (with full path if required) with the 
        master image (that that will be taken as a reference)
    slave: str 
        A filename (with path if needed) with the image
        that will be reprojected
    res: float, optional
        The desired output spatial resolution, if different 
        to the one in ``master``.
        
    Returns
    ----------
    The reprojected filename
    TODO Have a way of controlling output filename
    """

        
    slave_ds = gdal.Open( slave )
    if slave_ds is None:
        raise IOError, "GDAL could not open slave file %s " \
            % slave
    if s_srs is None:
        proj = slave_ds.GetProjection()
    else:
        proj = s_srs
     
         
    slave_geotrans = slave_ds.GetGeoTransform()
    if box is None:
        ulx = slave_geotrans[0]
        uly = slave_geotrans[3]
        lrx = slave_geotrans[0] + g.RasterXSize*slave_geotrans[1]
        lry = slave_geotrans[3] + g.RasterySize*slave_geotrans[-1]
    else:
        ulx, uly, lrx, lry = box
    if res is None:
        res = slave_geotrans[1]
    
        
    
    data_type = slave_ds.GetRasterBand(1).DataType
    n_bands = slave_ds.RasterCount
    
    if t_srs is None:
        master_proj = proj
    else:
        master_proj = t_srs
    
    master_geotrans = [ ulx, res, slave_geotrans[2], \
                uly, slave_geotrans[4], -res ]
    w = int((lrx - ulx)/res)
    h = int((uly - lry)/res)
    
    dst_filename= slave.replace( ".TIF", "_crop.tif" )
    if dst_filename == slave:
        dst_filename= slave.replace( ".tif", "_crop.tif" )
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, \
        w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( master_geotrans )
    dst_ds.SetProjection( proj )

    
    gdal.ReprojectImage( slave_ds, dst_ds, proj, \
        master_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None # Flush to disk
    return dst_filename 


class Spectral ( object ):
    def __init__ ( self, b_min, b_max ):
        self.b_min = b_min
        self.b_max = b_max
        
        wv = np.arange ( 400, 2501 )
        self.n_bands = b_min.shape[0]
        self.band_pass = np.zeros(( self.n_bands,2101), dtype=np.bool)
        self.bw = np.zeros( self.n_bands )
        bh = np.zeros( self.n_bands )
        for i in xrange( self.n_bands ):
            self.band_pass[i,:] = np.logical_and ( wv >= self.b_min[i], \
                    wv <= self.b_max[i] )
            self.bw[i] = self.b_max[i] - self.b_min[i]
        
class ObservationStorage ( object ):
    def __init__ ( self, datadir, resample_opts=None ):
        pass 
    
    def _setup_sensor ( self ):
        pass
    
    def _parse_metadata ( self ):
        pass

    def _get_emulators ( self ):
        pass

    def _sort_data ( self, resample_opts ):
        pass
    
    def get_data ( self ):
        pass
    
    def get_mask ( self ):
        pass
    
    def _get_emulators ( self ):
        pass
    
    def loop_observations ( self ):
        pass

class ETMObservations ( ObservationStorage ):
    """A class to locate and process ETM+ observations fro disk."""
    def __init__ ( self, datadir, resample_opts=None ):
        """The class takes the directory where the files sit. We expect to
        find an XML file with the metadata.
        """
        if not os.path.exists ( datadir ):
            raise IOError, "%s does not appear to exist in the filesystem?!"

        self.metadata = []
        for root, dirnames, filenames in os.walk( datadir ):
            for filename in fnmatch.filter(filenames, '*.xml'):
                self.metadata.append(os.path.join(root, filename))

        self.datadir = datadir


        if len ( self.metadata ) < 0:
            raise IOError, "No xml metadatafiles in %s" % self.datadir
        
        self._setup_sensor()
        self._parse_metadata ()
        self._get_emulators ()
        self._sort_data ( resample_opts )
        
    def _setup_sensor ( self ):
        """Sets up spectral stuff for this sensor."""
        self.spectral = Spectral ( np.array([450,520,630,770., 1550, 2090.] ), \
                    np.array([ 520, 600, 690, 900., 1750., 2350.] ) )                          
        
    def _parse_metadata ( self ):
        """Parse the metadata file. This assumes we have SPOT/Take5 XML files, but
        something similar will be available for Landsat or what not."""
        self.date =  []
        self.atcorr_refl = []
        self.saa = []
        self.sza = []
        self.vaa = []
        self.vza = []
        self.res = []
        self._mask = []
        for md_file in self.metadata:
            # This is required to get rid of the namespace cruft
            it = ET.iterparse ( md_file )
            for _, el in it:
                el.tag = el.tag.split('}', 1)[1]  # strip all namespaces
            tree = it.root

            dirname = os.path.dirname ( md_file )
            
            self.date.append(  datetime.datetime.strptime( \
                    tree.find("global_metadata/acquisition_date").text, \
                        "%Y-%m-%d") )
            
            for c in tree.findall ("global_metadata/corner"):
                if c.attrib['location'] == "UL":
                    ulx = float ( c.attrib['longitude'] )
                    uly = float ( c.attrib['latitude'] )
                else:
                    lrx = float ( c.attrib['longitude'] )
                    lry = float ( c.attrib['latitude'] )

                
            
            self.vaa.append ( get_vaa ( lrx, lry, ulx, uly )    )

            #self.atcorr_refl.append( os.path.join ( dirname, tree[1][2].text ) )
            self.saa.append( float (  root.find("global_metadata/solar_angles").attrib['azimuth'] ) )
            self.sza.append( float (  root.find("global_metadata/solar_angles").attrib['zenith'] ) )
            self.vza.append( 0.0 ) # Note that LDCM can look sideways a bit!
            self.res.append( 30. ) # 30m
            
            images = []
            mask = []
            for b in tree.findall("bands/band"):
                if b.attrib['product'] == "toa_refl":
                    fname = b.find("file_name").text
                    if fname.find ( "qa.tif" ) < 0:
                        images.append ( os.path.join ( dirname, fname ) )
                elif b.attrib['product'] == "cfmask":
                    mask = os.path.join ( dirname, fname )
            # Create VRT?     
            subprocess.call (["gdalbuildvrt", "-separate", \
                os.path.join ( dirname, md_file.replace(".xml", "_crop.vrt" )) ] + images )
            self.atcorr_refl.append ( os.path.join ( dirname, \
                md_file.replace(".xml", "_crop.vrt" )) )
            self._mask.append( mask )

    def _sort_data ( self, resample_opts ):
        """Get a pointer to the actual data, but don't load it into memory."""
        self._data_pntr = []
        for refl_file in self.atcorr_refl:
            if os.path.exists ( os.path.join ( self.datadir, refl_file ) ):
                if resample_opts is None:
                    fname = os.path.join ( self.datadir, refl_file ) 
                else:
                    fname = reproject_cut ( os.path.join ( self.datadir, refl_file ), \
                        **resample_opts )
                self._data_pntr.append ( gdal.Open ( fname )                                        )
            else:
            
                raise IOError, "GDAL cannot open this file: %s" % ( os.path.join ( \
                    self.datadir, refl_file) )
        self.resample_opts = resample_opts
            
    def get_data ( self ):
        """Return the data as a Numpy array. TODO Add a subset mechanism?"""
        return self._data_pntr.ReadAsArray()

    def get_mask ( self, iloc ):
        """Calculates the mask from its different components (again, SPOT4/Take5 data
        assumed). Will probably need to be modified for other sensors, and clearly
        we are assuming we have a mask already, in TOA processing, this might be a 
        good place to apply a simple cloud/cloud shadow mask or something like that."""
        mask = self._mask[iloc]
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, mask ), \
                        **self.resample_opts )

        g = gdal.Open( the_mask )
        sat = g.ReadAsArray()
        m3 = sat == 0

        the_mask = mask.replace("SAT", "DIV")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ), \
                        **self.resample_opts )        

        g = gdal.Open( the_mask  )
        div = g.ReadAsArray()
        m1 = div == 0 
    
        the_mask = mask.replace("SAT", "NUA")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ), \
                        **self.resample_opts )        

        g = gdal.Open( the_mask )
        nua = g.ReadAsArray()
        m2 = np.logical_not ( np.bitwise_and ( nua, 1 ).astype ( np.bool ) )
        return m1 * m2 * m3
    
    def _get_emulators ( self, model="prosail", emulator_home=\
            "/home/ucfajlg/Data/python/eoldas_ng_notebooks/emus/" ):
        """Based on the geometry, get the emulators. What could be simpler?"""
        files = glob.glob("%s*.npz" % emulator_home)
        emulator_search_dict = {}
        for f in files:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[1]),
                                float(f.split("/")[-1].split("_")[2]), \
                                float(f.split("/")[-1].split("_")[3])] = f
        # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
        # Remove some weirdos...

        
        emu_keys = np.array( emulator_search_dict.keys() )
        self.emulator = []
        for i in xrange (len ( self.metadata ) ):
            e_sza = emu_keys[np.argmin (np.abs( emu_keys[:,0] - self.sza[i] )), 0]
            e_vza = emu_keys[np.argmin (np.abs( emu_keys[:,2] - self.vza[i] )), 2]
            e_saa = emu_keys[np.argmin (np.abs( emu_keys[:,2] - self.saa[i] )), 1]
            e_vaa = emu_keys[np.argmin (np.abs( emu_keys[:,3] - self.vaa[i] )), 3]
            print self.sza[i], e_sza, self.vza[i], e_vza, self.vaa[i], e_vaa, self.saa[i], e_saa
            the_emulator = "%.1f_%.1f_%.1f_%.1f_%s.npz" % ( e_sza, e_saa, e_vza, e_vaa, model )
            print "Using emulator %s" % os.path.join ( emulator_home, the_emulator )
            self.emulator.append ( gp_emulator.MultivariateEmulator \
                        ( dump=os.path.join ( emulator_home, the_emulator ) ) )




################################################################        
class SPOTObservations ( ObservationStorage ):
    """An object that stores raster observations, including links
    to the data, a mask (or link to the mask), retrieves angles and
    decides on an observation operator.
    
    This particular implementation is designed with SPOT4/Take5 data in mind
    but it is expected that a similar approach could be taken with e.g.
    Landsat or Sentinel2 data.
    """

    def __init__ ( self, datadir, resample_opts=None ):
        """The class takes the directory where the files sit. We expect to
        find an XML file with the metadata.
        """
        ObservationStorage.__init__ ( self, datadir, resample_opts=None )
        if not os.path.exists ( datadir ):
            raise IOError, "%s does not appear to exist in the filesystem?!"

        self.metadata = []
        for root, dirnames, filenames in os.walk( datadir ):
            for filename in fnmatch.filter(filenames, '*.xml'):
                self.metadata.append(os.path.join(root, filename))

        self.datadir = datadir


        if len ( self.metadata ) < 0:
            raise IOError, "No xml metadatafiles in %s" % self.datadir
        
        self._setup_sensor()
        self._parse_metadata ()
        self._get_emulators ()
        self._sort_data ( resample_opts )
        
    def _setup_sensor ( self ):
        """Sets up spectral stuff for this sensor."""
        self.spectral = Spectral ( np.array([500,610,780,1580.] ), \
                    np.array([590,680,890,1750.] ) )                          
        
    def _parse_metadata ( self ):
        """Parse the metadata file. This assumes we have SPOT/Take5 XML files, but
        something similar will be available for Landsat or what not."""
        self.date =  []
        self.atcorr_refl = []
        self.saa = []
        self.sza = []
        self.vaa = []
        self.vza = []
        self.res = []
        self._mask = []

        for md_file in self.metadata:
            tree = xml.etree.ElementTree.ElementTree ( file=md_file ).getroot()
            dirname = os.path.dirname ( md_file )
            try:
                self.date.append(  datetime.datetime.strptime(tree[0][1].text, "%Y-%m-%d %H:%M:%S") )
            except:
                self.date.append(  datetime.datetime.strptime(tree[0][1].text, "%Y-%m-%d %H:%M:%S.%f") )
            self.atcorr_refl.append( os.path.join ( dirname, tree[1][2].text ) )
            self.saa.append( float ( tree[4][10][0].text ) )
            self.sza.append( float ( tree[4][10][1].text ) )
            self.vaa.append( float ( tree[4][10][2].text ) )
            self.vza.append( float ( tree[4][10][3].text ) )
            self.res.append( float ( tree[2][1].text ) )
            self._mask.append( os.path.join ( dirname, tree[1][5].text ) )

    def _sort_data ( self, resample_opts ):
        """Get a pointer to the actual data, but don't load it into memory."""
        self._data_pntr = []
        for refl_file in self.atcorr_refl:
            if os.path.exists ( os.path.join ( self.datadir, refl_file ) ):
                if resample_opts is None:
                    fname = os.path.join ( self.datadir, refl_file ) 
                else:
                    fname = reproject_cut ( os.path.join ( self.datadir, refl_file ), \
                        **resample_opts )
                self._data_pntr.append ( gdal.Open ( fname )                                        )
            else:
            
                raise IOError, "GDAL cannot open this file: %s" % ( os.path.join ( \
                    self.datadir, refl_file) )
        self.resample_opts = resample_opts
            
    def get_data ( self ):
        """Return the data as a Numpy array. TODO Add a subset mechanism?"""
        return self._data_pntr.ReadAsArray()

    def get_mask ( self, iloc ):
        """Calculates the mask from its different components (again, SPOT4/Take5 data
        assumed). Will probably need to be modified for other sensors, and clearly
        we are assuming we have a mask already, in TOA processing, this might be a 
        good place to apply a simple cloud/cloud shadow mask or something like that."""
        mask = self._mask[iloc]
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, mask ), \
                        **self.resample_opts )

        g = gdal.Open( the_mask )
        sat = g.ReadAsArray()
        m3 = sat == 0

        the_mask = mask.replace("SAT", "DIV")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ), \
                        **self.resample_opts )        

        g = gdal.Open( the_mask  )
        div = g.ReadAsArray()
        m1 = div == 0 
    
        the_mask = mask.replace("SAT", "NUA")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ), \
                        **self.resample_opts )        

        g = gdal.Open( the_mask )
        nua = g.ReadAsArray()
        m2 = np.logical_not ( np.bitwise_and ( nua, 1 ).astype ( np.bool ) )
        return m1 * m2 * m3
    
    def _get_emulators ( self, model="prosail", emulator_home=\
            "/home/ucfajlg/Data/python/eoldas_ng_notebooks/emus/" ):
        """Based on the geometry, get the emulators. What could be simpler?"""
        files = glob.glob("%s*.npz" % emulator_home)
        emulator_search_dict = {}
        for f in files:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[1]),
                                float(f.split("/")[-1].split("_")[2]), \
                                float(f.split("/")[-1].split("_")[3])] = f
        # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
        # Remove some weirdos...

        
        emu_keys = np.array( emulator_search_dict.keys() )
        self.emulator = []
        for i in xrange (len ( self.metadata ) ):
            e_sza = emu_keys[np.argmin (np.abs( emu_keys[:,0] - self.sza[i] )), 0]
            e_vza = emu_keys[np.argmin (np.abs( emu_keys[:,2] - self.vza[i] )), 2]
            e_saa = emu_keys[np.argmin (np.abs( emu_keys[:,2] - self.saa[i] )), 1]
            e_vaa = emu_keys[np.argmin (np.abs( emu_keys[:,3] - self.vaa[i] )), 3]
            print self.sza[i], e_sza, self.vza[i], e_vza, self.vaa[i], e_vaa, self.saa[i], e_saa
            the_emulator = "%.1f_%.1f_%.1f_%.1f_%s.npz" % ( e_sza, e_saa, e_vza, e_vaa, model )
            print "Using emulator %s" % os.path.join ( emulator_home, the_emulator )
            self.emulator.append ( gp_emulator.MultivariateEmulator \
                        ( dump=os.path.join ( emulator_home, the_emulator ) ) )
            
    def loop_observations ( self, start_date, end_date, step=1, fmt="%Y-%m-%d" ):
        """This is a generator method that loops over the available 
        observations between a start and an end point with a given
        temporal resolution (in days)"""
        
        start_date = datetime.datetime.strptime( start_date, fmt )
        end_date = datetime.datetime.strptime( end_date, fmt )
        if start_date < self.date[0]:
            print "No observations until %s, starting from there" % self.date[0]
            start_date = self.date[0]
            
        if end_date > self.date[-1]:
            print "No observations after %s, stopping there" % self.date[-1]
            end_date = self.date[-1]
            
        delta = datetime.timedelta ( days=step )
        this_date = start_date.date()
        end_date = end_date.date() + delta
        obs_dates = [ x.date() for x in self.date ]
        while this_date < end_date:
            if this_date in obs_dates:
                iloc = obs_dates.index ( this_date )
                have_obs = True
                the_data = self._data_pntr[iloc].ReadAsArray()
                the_mask = self.get_mask ( iloc )
                the_emulator = self.emulator[ iloc ] 
                the_sza = self.sza[ iloc ]
                the_saa = self.saa[ iloc ]
                the_vza = self.vza[ iloc ]
                the_vaa = self.vaa[ iloc ]
                the_fname = self._data_pntr[iloc].GetDescription()
                the_spectrum = self.spectral
                
            else:
                have_obs = False
                the_data = None
                the_mask = None
                the_emulator = None
                the_sza = None
                the_saa = None
                the_vza = None
                the_vaa = None
                the_fname = None
                the_spectrum = None
            this_date += delta
            retval = namedtuple ( "retval", ["have_obs", "date", "image", "mask", "emulator", \
                "sza", "saa", "vza", "vaa", "fname", "spectrum"] )
            retvals = retval ( have_obs=have_obs, \
                date=this_date - delta, image=the_data, mask=the_mask, emulator=the_emulator, sza=the_sza, \
                saa=the_saa, vza=the_vza, vaa=the_vaa, fname=the_fname, spectrum=the_spectrum )
            yield retvals

        

        

    
class SingleImageProcessor ( object ):
    """This class sets up and solves a single date/single image using 
    eoldas_ng"""

    def __init__ ( self, state_config, state_grid, image, band_unc, mask, \
        bw, band_pass, emulator, prior, regularisation, process_name, \
        factor, optimisation_options=None, verbose=False ):
        """The class creator. Takes state configuration and grids (spatial),
        band uncertainties, image and associated mask, relevant emulator, a
        prior object and a regularisation (spatial) model. We also have a 
        process name to store the results more or less neatly.

        TODO We ought to have private methods to set up the different 
        objects, as that allows the user to select a different type of
        "pre-packaged" state (e.g. not the prosail one).
        
        Parameters
        -----------
        state_config: dict
            A state configuration object
        state_grid: array
            A state grid (2D)
        image: array
            A multispectral image (3D)
        band_unc: array
            Per band uncertainty. Note that I need to check whether you
            can have a per pixel uncertainty, or whether that requires
            some extra recoding. CHECK
        mask: array
            A 2D array, indicating pixels which are OK and pixels which 
            aren't
        bw: array?
            A BW array. This is needed for somem reason, but i think it's
            superfluous CHECK
        band_pass: list
            A list of band_pass objects. We use this list (and the ``bw``
            parameter above) to calculate per band emulators and the 
            initial inverse emulators.
        emulator: list
            The emulator for this particular image acquisition geometry.
            In principle this is a MultivariateEmulator in a single element 
            list, but could be per band emulators
        prior: Prior
            An ``eoldas_ng`` prior object
        regularisation: SpatialSmoother
            An ``eoldas_ng`` SpatialSmoother object.
        process_name: str 
            A string with the process name. This is used to store the 
            results of inverting the current image
        factor: int
            The spatial factor: how many times does this observation
            fit in with the state spatial resolution?
        optimisation_options: dict
            A dictionary of optimisation options
        verbose: bool
            A verbose flag            
        """
        
        # Get the state [CENSORED]
        self.the_state = StandardStatePROSAIL ( state_config, state_grid, \
                 optimisation_options=optimisation_options, \
                 output_name=process_name, verbose=verbose )
        self._build_observation_constraint ( state_grid, image, mask, 
                                            emulator, band_unc, factor,
                                            band_pass, bw )
        # Note that this isn't very satisfactory, but we can always change 
        # that: the prior just needs a mean and a sparse covariance, and 
        # the regularisation needs the gamma(s) and parameters to apply it
        # to
        self.the_prior = prior
        self.the_model = regularisation
        print "WARNING! No prior involved here!"
        print "PRIOR needs defining!"
        #self.the_state.add_operator ( "Prior", self.the_prior )
        self.the_state.add_operator ( "Regularisation", self.the_model )
        self.the_state.add_operator ( "Observations", self.the_observations )
        
    def _build_observation_constraint ( self, state_grid, image, image_mask, \
        emulator, band_uncertainty, factor, band_pass, bw, per_band=True):
        """This is just a method to build up the observation operator."""
        
        self.the_observations = ObservationOperatorImageGP ( state_grid, \
                self.the_state, image, image_mask, emulator, \
                bu=band_uncertainty, factor=factor, \
                band_pass=band_pass, bw=bw, per_band=per_band )
    def first_guess ( self ):
        
        return self.the_observations.first_guess ( \
            self.the_state.state_config )

    def solve ( self, x0=None ):
        return self.the_state.optimize ( x0=x0, do_unc=True )
    
            
"""Some requirements for processing:
* We need a a state config, both for each image and temporally, as we might 
need to have parametes with no temporal correlation (atmosphere). So we need
a ``time_state_config`` and ``space_state_config``. The complication here is
that the Hessian needs to be modified to revert to the prior for the 
parameters that do not vary in time, which is a pain in the arse.

* There can be more than ons observation per time step. This helps with e.g.
atmospheric characterisation

* We might need to rearrange the individual posterior state dumps in some
easy to use/visualise GDAL geo files.

* I can't quite remember the update equation for the Hessian ;-)
Here it is: If we have the posterior Hessian, $\mathbf{A}^{-1}$, and the inflation 
matrix, $\mathbf{B}$, then the resulting "inflated" matrix is
$$
\begin{align}
\mathbf{C}_{prior}^{-1}&=(\mathbf{A} + \mathbf{B})^{-1} \\
    =& \mathbf{A}^{-1}-\mathbf{A}^{-1}\mathbf{B}\left[\mathbf{I}+\mathbf{A}^{-1}\mathbf{B}\right]^{-1}\mathbf{A}^{-1}\\
\end{align}
$$

or, in python parlance...
a_i - a_i.dot(b).dot ( np.linalg.inv(np.eye(5) + a_i.dot(b))).dot(a_i)

How does this work with a sparse matrix is still open to question...

"""

class SpaceTimeProcessor ( object ):

    def __init__ ( self, start_time, end_time, time_resolution, space_grid, \
            observations, space_gamma, time_gamma ):
        
        self.time_grid = time_grid
        self.space_grid = space_grid
        self.observations = observations
        self.space_gamma = space_gamma
        self.time_gamma = time_gamma
        
    def initialise_process ( self ):
        """This method initialises the entire process """
        
        self.prior = set_prior (state, prev_date=None )
        self.smoother = SpatialSmoother ( self.space_grid, self.space_gamma, \
            required_params=["lai", "cab", "cw", "cm", "psoil", "bsoil"] )

    def inflate_uncertainty ( hessian ):
        N = a_i.shape[0]
        # b here is inflation matrix, should just be diagonal, right?
        return hessian - hessian.dot(b).dot ( np.linalg.inv(sp.eye(N) + a_i.dot(b))).dot(a_i)
    
if __name__ == "__main__":
#    obs1 = ObservationStorage ( "/storage/ucfajlg/MidiPyrenees")
    state_grid = np.zeros ( ( 584, 349 ) )
    
    state_config = OrderedDict ()
    state_config['n'] = FIXED
    state_config['cab'] = VARIABLE
    state_config['car'] = FIXED
    state_config['cbrown'] = FIXED
    state_config['cw'] = VARIABLE
    state_config['cm'] = VARIABLE
    state_config['lai'] = VARIABLE
    state_config['ala'] = FIXED
    state_config['bsoil'] = VARIABLE
    state_config['psoil'] = VARIABLE

    spatial_factor = [1,1]
    space_gamma = 0.5
    the_smoother = SpatialSmoother ( state_grid, space_gamma, \
            required_params=["lai", "cab", "cw", "cm", "psoil", "bsoil"] )
#    the_prior = set_prior ()
    
    resample_opts = {'box': [546334.113775153,  6274489.49408634,  \
        558032.21126551,  6267491.58431377]}
    bu = [ 0.02, 0.02, 0.02, 0.02]
    spot_observations = SPOTObservations( "/storage/ucfajlg/MidiPyrenees/SPOT", resample_opts )
    for s in spot_observations.loop_observations ( "2013-01-01", "2013-12-31" ):
        if s.have_obs: 
            print s.date
            # First element of iterator is True, then we have an observation
            # What is needed at this stage is
            # 1. The prior object (see ``set_prior`` function)
            # 2. The spatial smoother
            # 3. A state configuration dictionary
            # 4. A state grid
            # 5. The spatial factor (ratio of observation spatial resolution
            #    to state spatial resolution. Must be integer)
            # 6. The actual data array 
            # 7. The relevant mask
            # 8. The observational uncertainty
            # 9. Spectral parameters
            # 10. Process name from the actual image filename
            this_image = SingleImageProcessor ( state_config, state_grid,
                                               s.image/1000., bu, s.mask, s.spectrum.bw,
                                               s.spectrum.band_pass, s.emulator, the_smoother, 
                                               the_smoother, s.fname.replace("_crop.tif", ""),
                                               spatial_factor )
            x0 = this_image.first_guess ()
            
            break
                ###def __init__ ( self, state_config, state_grid, image, band_unc, mask, \
        ###bw, band_pass, emulator, prior, regularisation, process_name, \
        ###factor, optimisation_options=None, verbose=False ):

            