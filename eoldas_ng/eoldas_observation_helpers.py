import subprocess
import fnmatch
import datetime
import glob
import os
import xml.etree.ElementTree
from collections import namedtuple

from osgeo import gdal
import numpy as np

import gp_emulator

FIXED = 1
CONSTANT = 2
VARIABLE = 3


def get_vaa ( lrx, lry, ulx, uly ):

    dx = ulx - lrx
    dy = uly - lry
    m = np.sqrt ( dx * dx + dy * dy )
    r = np.atan2 ( dx / m, dy / m )
    d = -180. * r / np.pi
    if d < 0:
        d = -d


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

    dst_filename = slave.replace( ".tif", "_crop.vrt" )
    dst_ds = gdal.GetDriverByName('VRT').Create(dst_filename,
                                                w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( master_geotrans )
    dst_ds.SetProjection( master_proj)

    gdal.ReprojectImage( slave_ds, dst_ds, slave_proj,
                         master_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None  # Flush to disk
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
        lrx = slave_geotrans[0] + slave_ds.RasterXSize * slave_geotrans[1]
        lry = slave_geotrans[3] + slave_ds.RasterySize * slave_geotrans[-1]
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

    master_geotrans = [ ulx, res, slave_geotrans[2],
                        uly, slave_geotrans[4], -res ]
    w = int((lrx - ulx) / res)
    h = int((uly - lry) / res)

    dst_filename = slave.replace( ".TIF", "_crop.tif" )
    if dst_filename == slave:
        dst_filename = slave.replace( ".tif", "_crop.tif" )
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename,
                                                  w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( master_geotrans )
    dst_ds.SetProjection( proj )

    gdal.ReprojectImage( slave_ds, dst_ds, proj,
                         master_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None  # Flush to disk
    return dst_filename


class Spectral ( object ):

    def __init__ ( self, b_min, b_max ):
        self.b_min = b_min
        self.b_max = b_max

        wv = np.arange ( 400, 2501 )
        self.n_bands = b_min.shape[0]
        self.band_pass = np.zeros(( self.n_bands, 2101), dtype=np.bool)
        self.bw = np.zeros( self.n_bands )
        self.bh = np.zeros( self.n_bands )
        for i in xrange( self.n_bands ):
            self.band_pass[i, :] = np.logical_and ( wv >= self.b_min[i],
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
        self.spectral = Spectral ( np.array([450, 520, 630, 770., 1550, 2090.] ),
                                   np.array([ 520, 600, 690, 900., 1750., 2350.] ) )

    def _parse_metadata ( self ):
        """Parse the metadata file. This assumes we have SPOT/Take5 XML files, but
        something similar will be available for Landsat or what not."""
        self.date = []
        self.atcorr_refl = []
        self.saa = []
        self.sza = []
        self.vaa = []
        self.vza = []
        self.res = []
        self._mask = []
        for md_file in self.metadata:
            # This is required to get rid of the namespace cruft
            it = xml.etree.ElementTree.iterparse ( md_file )
            for _, el in it:
                el.tag = el.tag.split('}', 1)[1]  # strip all namespaces
            tree = it.root

            dirname = os.path.dirname ( md_file )

            self.date.append(  datetime.datetime.strptime(
                tree.find("global_metadata/acquisition_date").text,
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
            self.saa.append(
                float (  tree.find("global_metadata/solar_angles").attrib['azimuth'] ) )
            self.sza.append(
                float (  tree.find("global_metadata/solar_angles").attrib['zenith'] ) )
            self.vza.append( 0.0 )  # Note that LDCM can look sideways a bit!
            self.res.append( 30. )  # 30m

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
            subprocess.call (["gdalbuildvrt", "-separate",
                              os.path.join ( dirname, md_file.replace(".xml", "_crop.vrt" )) ] + images )
            self.atcorr_refl.append ( os.path.join ( dirname,
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
                    fname = reproject_cut ( os.path.join ( self.datadir, refl_file ),
                                            **resample_opts )
                self._data_pntr.append (
                    gdal.Open ( fname )                                        )
            else:

                raise IOError, "GDAL cannot open this file: %s" % ( os.path.join (
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
            the_mask = reproject_cut ( os.path.join ( self.datadir, mask ),
                                       **self.resample_opts )

        g = gdal.Open( the_mask )
        sat = g.ReadAsArray()
        m3 = sat == 0

        the_mask = mask.replace("SAT", "DIV")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ),
                                       **self.resample_opts )

        g = gdal.Open( the_mask  )
        div = g.ReadAsArray()
        m1 = div == 0

        the_mask = mask.replace("SAT", "NUA")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ),
                                       **self.resample_opts )

        g = gdal.Open( the_mask )
        nua = g.ReadAsArray()
        m2 = np.logical_not ( np.bitwise_and ( nua, 1 ).astype ( np.bool ) )
        return m1 * m2 * m3

    def _get_emulators ( self, model="prosail", emulator_home="/home/ucfajlg/Data/python/eoldas_ng_notebooks/emus/" ):
        """Based on the geometry, get the emulators. What could be simpler?"""
        files = glob.glob("%s*.npz" % emulator_home)
        emulator_search_dict = {}
        for f in files:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]),
                                  float(f.split("/")[-1].split("_")[1]),
                                  float(f.split("/")[-1].split("_")[2]),
                                  float(f.split("/")[-1].split("_")[3])] = f
        # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
        # Remove some weirdos...

        emu_keys = np.array( emulator_search_dict.keys() )
        self.emulator = []
        for i in xrange (len ( self.metadata ) ):
            e_sza = emu_keys[
                np.argmin (np.abs( emu_keys[:, 0] - self.sza[i] )), 0]
            e_vza = emu_keys[
                np.argmin (np.abs( emu_keys[:, 2] - self.vza[i] )), 2]
            e_saa = emu_keys[
                np.argmin (np.abs( emu_keys[:, 2] - self.saa[i] )), 1]
            e_vaa = emu_keys[
                np.argmin (np.abs( emu_keys[:, 3] - self.vaa[i] )), 3]
            print self.sza[i], e_sza, self.vza[i], e_vza, self.vaa[i], e_vaa, self.saa[i], e_saa
            the_emulator = "%.1f_%.1f_%.1f_%.1f_%s.npz" % (
                e_sza, e_saa, e_vza, e_vaa, model )
            print "Using emulator %s" % os.path.join ( emulator_home, the_emulator )
            self.emulator.append ( gp_emulator.MultivariateEmulator
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
        self.spectral = Spectral ( np.array([500, 610, 780, 1580.] ),
                                   np.array([590, 680, 890, 1750.] ) )

    def _parse_metadata ( self ):
        """Parse the metadata file. This assumes we have SPOT/Take5 XML files, but
        something similar will be available for Landsat or what not."""
        self.date = []
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
                self.date.append(
                    datetime.datetime.strptime(tree[0][1].text, "%Y-%m-%d %H:%M:%S") )
            except:
                self.date.append(
                    datetime.datetime.strptime(tree[0][1].text, "%Y-%m-%d %H:%M:%S.%f") )
            self.atcorr_refl.append(
                os.path.join ( dirname, tree[1][2].text ) )
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
                    fname = reproject_cut ( os.path.join ( self.datadir, refl_file ),
                                            **resample_opts )
                self._data_pntr.append (
                    gdal.Open ( fname )                                        )
            else:

                raise IOError, "GDAL cannot open this file: %s" % ( os.path.join (
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
            the_mask = reproject_cut ( os.path.join ( self.datadir, mask ),
                                       **self.resample_opts )

        g = gdal.Open( the_mask )
        sat = g.ReadAsArray()
        m3 = sat == 0

        the_mask = mask.replace("SAT", "DIV")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ),
                                       **self.resample_opts )

        g = gdal.Open( the_mask  )
        div = g.ReadAsArray()
        m1 = div == 0

        the_mask = mask.replace("SAT", "NUA")
        if self.resample_opts is not None:
            the_mask = reproject_cut ( os.path.join ( self.datadir, the_mask ),
                                       **self.resample_opts )

        g = gdal.Open( the_mask )
        nua = g.ReadAsArray()
        m2 = np.logical_not ( np.bitwise_and ( nua, 1 ).astype ( np.bool ) )
        return m1 * m2 * m3

    def _get_emulators ( self, model="prosail", emulator_home="/home/ucfajlg/Data/python/eoldas_ng_notebooks/emus/" ):
        """Based on the geometry, get the emulators. What could be simpler?"""
        files = glob.glob("%s*.npz" % emulator_home)
        emulator_search_dict = {}
        for f in files:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]),
                                  float(f.split("/")[-1].split("_")[1]),
                                  float(f.split("/")[-1].split("_")[2]),
                                  float(f.split("/")[-1].split("_")[3])] = f
        # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
        # Remove some weirdos...

        emu_keys = np.array( emulator_search_dict.keys() )
        self.emulator = []
        for i in xrange (len ( self.metadata ) ):
            e_sza = emu_keys[
                np.argmin (np.abs( emu_keys[:, 0] - self.sza[i] )), 0]
            e_vza = emu_keys[
                np.argmin (np.abs( emu_keys[:, 2] - self.vza[i] )), 2]
            e_saa = emu_keys[
                np.argmin (np.abs( emu_keys[:, 2] - self.saa[i] )), 1]
            e_vaa = emu_keys[
                np.argmin (np.abs( emu_keys[:, 3] - self.vaa[i] )), 3]
            print self.sza[i], e_sza, self.vza[i], e_vza, self.vaa[i], e_vaa, self.saa[i], e_saa
            the_emulator = "%.1f_%.1f_%.1f_%.1f_%s.npz" % (
                e_sza, e_saa, e_vza, e_vaa, model )
            print "Using emulator %s" % os.path.join ( emulator_home, the_emulator )
            self.emulator.append ( gp_emulator.MultivariateEmulator
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
            retval = namedtuple ( "retval", ["have_obs", "date", "image", "mask", "emulator",
                                             "sza", "saa", "vza", "vaa", "fname", "spectrum"] )
            retvals = retval ( have_obs=have_obs,
                               date=this_date - delta, image=the_data, mask=the_mask, emulator=the_emulator, sza=the_sza,
                               saa=the_saa, vza=the_vza, vaa=the_vaa, fname=the_fname, spectrum=the_spectrum )
            yield retvals
