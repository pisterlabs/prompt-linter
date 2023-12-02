import numpy as np
import os
import cohere_core as cohere
import util.util as ut

class Detector(object):
    """
    class Detector(self)
    ===========================

    Abstract class representing detector.

    """
    __all__ = ['get_frame',
               'insert_seam',
               'clear_seam',
               'get_pixel'
               ]

    def __init__(self):
        self.name = "default"

    def get_frame(self, filename, roi, Imult):
        """
        Reads raw 2D frame from a file. Concrete function in subclass applies correction for the specific detector. For example it could be darkfield correction or whitefield correction.

        Parameters
        ----------
        filename : str
            data file name
        roi : list
            detector area used to take image. If None the entire detector area will be used.
        Imult : int
            multiplier

        Returns
        -------
        ndarray
            frame after instrument correction

        """
        filename = filename.replace(os.sep, '/')
        self.raw_frame = ut.read_tif(filename)
        return self.raw_frame

    def insert_seam(self, arr, roi=None):
        """
        Corrects the non-continuous areas of detector. Concrete function in subclass inserts rows/columns in frame as instrument correction.

        Parameters
        ----------
        arr : ndarray
            frame to insert the correction
        roi : list
            detector area used to take image. If None the entire detector area will be used.

        Returns
        -------
        ndarray
            frame after instrument correction

        """
        return arr

    def clear_seam(self, arr):
        """
        Corrects the non-continuous areas of detector. Concrete function in subclass removes rows/columns in frame as instrument correction.

        Parameters
        ----------
        arr : ndarray
            frame to apply the correction
        roi : list
            detector area used to take image. If None the entire detector area will be used.

        Returns
        -------
        ndarray
            frame after instrument correction

        """
        return arr


    def set_detector(self, conf_map):
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # if anything in config file has the same name as a required detector attribute, copy it to
        # the detector
        # this will capture things like whitefield_filename, etc.
        for key, val in conf_map.items():
            if hasattr(self, key):
                setattr(self, key, val)


    def get_pixel(self):
        """
        Returns detector pixel size.  Concrete function in subclass returns value applicable to the detector.

        Returns
        -------
        tuple
            size of pixel

        """
        pass


class Detector_34idcTIM1(Detector):
    """
    Subclass of Detector. Encapsulates "34idcTIM1" detector.
    """
    name = "34idcTIM1"
    dims = (256, 256)
    roi = (256, 256)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    darkfield_filename = None
    darkfield = None
    raw_frame = None


    def __init__(self):
        super(Detector_34idcTIM1, self).__init__()


    def load_darkfield(self):
        """
        Reads darkfield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.darkfield = cohere.read_tif(self.darkfield_filename)
        except:
            print("Darkfield filename not set for TIM1, will not correct")


    def get_raw_frame(self, filename):
        try:
            self.raw_frame = cohere.read_tif(filename)
        except:
            print("problem reading raw file ", filename)
            raise


    # TIM1 only needs bad pixels deleted.  Even that is optional.
    def get_frame(self, filename, roi, Imult):
        """
        Reads raw frame from a file, and applies correction for 34idcTIM1 detector, i.e. darkfield.
        Parameters
        ----------
        filename : str
            data file name
        roi : list
            detector area used to take image. If None the entire detector area will be used.
            
        Imult : float
            value to fill the sem with
        Returns
        -------
        frame : ndarray
            frame after correction
        """
        if roi is None:
            roi = (0, 256, 0, 256)
        if Imult is None:
            Imult = 1.0
        if not type(self.darkfield) == np.ndarray:
            self.load_darkfield()

        roislice1 = slice(roi[0], roi[0] + roi[1])
        roislice2 = slice(roi[2], roi[2] + roi[3])

        self.get_raw_frame(filename)
        try:
            frame = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, self.raw_frame)
        except:
            frame = self.raw_frame

        return frame


    def get_missing_attr(self):
        # The TIM1 detector does not require whitefield or darkfield for correction
        return None


    def get_pixel(self):
        """
        Returns pixel size of 34idcTIM1 detector.
        Parameters
        ----------
        none
        Returns
        -------
        tuple
            size of pixel
        """
        return self.pixel


class Detector_34idcTIM2(Detector):
    """
    Subclass of Detector. Encapsulates "34idcTIM2" detector.
    """
    name = "34idcTIM2"
    dims = (512, 512)
    roi = (0, 512, 0, 512)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    whitefield_filename = None
    darkfield_filename = None
    whitefield = None
    darkfield = None
    raw_frame = None


    def __init__(self):
        super(Detector_34idcTIM2, self).__init__()


    def load_whitefield(self):
        """
        Reads whitefield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.whitefield = cohere.read_tif(self.whitefield_filename)
        except:
            print("Whitefield filename not set for TIM2")
            raise
        try:
#            self.whitefield = np.where(self.whitefield < 100, 1e20, self.whitefield) #Some large value
            self.whitefield[255:257,0:255] = 0   #wierd pixels on edge of seam (TL/TR). Kill in WF kills in returned frame as well.
            self.wfavg=np.average(self.whitefield)
            self.wfstd=np.std(self.whitefield)
            self.whitefield = np.where( self.whitefield < self.wfavg-3*self.wfstd, 0, self.whitefield)
        except:
            print("Corrections to the TIM2 whitefield image failed in detector module.")


    def load_darkfield(self):
        """
        Reads darkfield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.darkfield = cohere.read_tif(self.darkfield_filename)
        except:
            print("Darkfield filename not set for TIM2")
            raise
        if type(self.whitefield) == np.ndarray:
            self.whitefield = np.where(self.darkfield > 1, 0, self.whitefield) #kill known bad pixel

    def get_raw_frame(self, filename):
        try:
            self.raw_frame = cohere.read_tif(filename)
        except:
            print("problem reading raw file ", filename)
            raise


    def get_frame(self, filename, roi, Imult):
        """
        Reads raw frame from a file, and applies correction for 34idcTIM2 detector, i.e. darkfield, whitefield, and
seam.
        Parameters
        ----------
        filename : str
            data file name
        roi : list
            detector area used to take image. If None the entire detector area will be used.
            
        Imult : float
            value to fill the sem with
        Returns
        -------
        frame : ndarray
            frame after correction
        """
        # roi is start,size,start,size
        # will be in imageJ coords, so might need to transpose,or just switch x-y
        # divide whitefield
        # blank out pixels identified in darkfield
        # insert 4 cols 5 rows if roi crosses asic boundary
        if roi is None:
            roi = (0, 512, 0, 512)
        #WFprocessing using darkfield.  So do it first.
        if not type(self.darkfield) == np.ndarray:
            self.load_darkfield()
        if not type(self.whitefield) == np.ndarray:
            self.load_whitefield()
        if Imult is None:
            Imult = self.wfavg

        roislice1 = slice(roi[0], roi[0] + roi[1])
        roislice2 = slice(roi[2], roi[2] + roi[3])

        # some of this should probably be in try blocks
        self.get_raw_frame(filename)
        normframe = self.raw_frame / self.whitefield[roislice1, roislice2] * Imult
        normframe = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, normframe)
        normframe = np.where(np.isfinite(normframe), normframe, 0)
        frame = self.insert_seam(normframe, roi)
        frame = np.where(np.isnan(frame),0,frame)
        return frame
        

    # frame here can also be a 3D array.
    def insert_seam(self, arr, roi):
        """
        Inserts rows/columns correction in a frame for 34idcTIM2 detector.
        Parameters
        ----------
        arr : ndarray
            raw frame
        roi : list
            detector area used to take image. If None the entire detector area will be used.
        Returns
        -------
        frame : ndarray
            frame after insering rows/columns
        """
        # Need to break this out.  When aligning multi scans the insert will mess up the aligns
        # or maybe we just need to re-blank the seams after the aligns?
        # I can't decide if the seams are a detriment to the alignment.  might need to try some.
        s1range = range(roi[0], roi[0] + roi[1])
        s2range = range(roi[2], roi[2] + roi[3])
        dims = arr.shape

        # get the col that start at det col 256 in the roi
        try:
            i1 = s1range.index(256)  # if not in range try will except
            if i1 != 0:
                frame = np.insert(arr, i1, np.zeros((4, dims[0])), axis=0)
            # frame=np.insert(normframe, i1, np.zeros((5,dims[0])),axis=0)
            else:
                frame = arr
        except:
            frame = arr  # if there's no insert on dim1 need to copy to frame

        try:
            i2 = s2range.index(256)
            if i2 != 0:
                frame = np.insert(frame, i2, np.zeros((5, dims[0] + 4)), axis=1)
        except:
            # if there's no insert on dim2 thre's nothing to do
            pass

        return frame


    # This is needed if the seam has already been inserted and shifts have moved intensity
    # into the seam.  Found that alignment of data sets was best done with the seam inserted.
    # For instance.
    def clear_seam(self, arr):
        """
        Removes rows/columns correction from a frame for 34idcTIM2 detector.
        Parameters
        ----------
        arr : ndarray
            frame to remove seam
        roi : list
            detector area used to take image. If None the entire detector area will be used.
        Returns
        -------
        arr : ndarray
            frame after removing rows/columns
        """
        # modify the slices if 256 is in roi
        roi = self.roi
        s1range = range(roi[0], roi[0] + roi[1])
        s2range = range(roi[2], roi[2] + roi[3])
        try:
            i1 = s1range.index(256)  # if not in range try will except
            if i1 != 0:
                s1range[0] = slice(i1, i1 + 4)
                arr[tuple(s1range)] = 0
        except:
            pass
            #print("no clear on dim0")
        try:
            i2 = s2range.index(256)
            if i2 != 0:
                s2range[1] = slice(i2, i2 + 5)
                arr[tuple(s2range)] = 0
        except:
            pass
            #print("no clear on dim1")
            
        return arr


    def get_pixel(self):
        """
        Returns pixel size of 34idcTIM2 detector.
        Parameters
        ----------
        none
        Returns
        -------
        tuple
            size of pixel
        """
        return self.pixel


def create_detector(det_name):
    if det_name == '34idcTIM1':
        return Detector_34idcTIM1()
    elif det_name == '34idcTIM2':
        return Detector_34idcTIM2()
    else:
        print ('detector ' + det_name + ' not defined.')
        return None


