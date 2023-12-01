import os
import sys
import re
import glob
import numpy as np
from xrayutilities.io import spec as spec
from multiprocessing import Pool, Process, cpu_count
import cohere_core as cohere
import util.util as ut


def get_det_from_spec(specfile, scan, **kwargs):
    """
    Reads detector area and detector name from spec file for given scan.
    Parameters
    ----------
    specfile : str
        spec file name
         
    scan : int
        scan number to use to recover the saved measurements
    Returns
    -------
    detector_name : str
        detector name
    det_area : list
        detector area
    """
    try:
    # Scan numbers start at one but the list is 0 indexed
        ss = spec.SPECFile(specfile)[scan - 1]
    # Stuff from the header
        detector_name = str(ss.getheader_element('UIMDET'))
        det_area = [int(n) for n in ss.getheader_element('UIMR5').split()]
        return detector_name, det_area
    except  Exception as ex:
        print(str(ex))
        print ('Could not parse ' + specfile)
        return None, None


class BeamPrepData():
    """
    This class contains fields needed for the data preparation, parsed from spec or configuration file.
    The class uses helper functions to prepare the data.
    """

    def __init__(self, experiment_dir, main_conf_map, prep_conf_map, *args, **kwargs):
        """
        Creates PrepData instance for beamline aps_34idc. Sets fields to configuration parameters.
        Parameters
        ----------
        experiment_dir : str
            directory where the files for the experiment processing are created
        Returns
        -------
        PrepData object
        """
        self.printed_dims = False
        self.args = args
        self.experiment_dir = experiment_dir

        self.det_name = None
        self.roi = None
        self.scan_ranges = []
        if 'scan' in main_conf_map:
            scan_units = [u for u in main_conf_map['scan'].replace(' ','').split(',')]
            for u in scan_units:
                if '-' in u:
                    r = u.split('-')
                    self.scan_ranges.append([int(r[0]), int(r[1])])
                else:
                    self.scan_ranges.append([int(u), int(u)])
            scan_end = self.scan_ranges[-1][-1]
        else:
            print("scans not defined in main config")
            scan_end = None
        if scan_end is not None:
            if 'specfile' in main_conf_map:
                specfile = main_conf_map['specfile']
                # parse det name and saved roi from spec
                try:
                    self.det_name, self.roi = get_det_from_spec(specfile, scan_end)
                except:
                    print("exception parsing spec file")
                if self.det_name is not None and self.det_name.endswith(':'):
                    self.det_name = self.det_name[:-1]
            else:
                print("specfile not configured")

        # detector name from configuration will override the one passed from spec file
        if 'detector' in prep_conf_map:
            self.det_name = prep_conf_map['detector']
        else:
            if self.det_name is None:
                # default detector get_frame method just reads tif files and doesn't do anything to them.
                print('Detector name is not available, using default detector class')
                self.det_name = "default"

        # if roi is in config file, use it, just in case spec had it wrong or it's not there.
        try:
            self.roi = prep_conf_map['roi']
        except:
            pass

        try:
            self.separate_scans = prep_conf_map['separate_scans']
        except:
            self.separate_scans = False

        try:
            self.separate_scan_ranges = prep_conf_map['separate_scan_ranges']
        except:
            self.separate_scan_ranges = False

        try:
            self.Imult = prep_conf_map['Imult']
        except:
            self.Imult = None

        try:
            self.min_files = self.prep_map['min_files']
        except:
            self.min_files = 0
        try:
            self.exclude_scans = self.prep_map['exclude_scans']
        except:
            self.exclude_scans = []


    def get_dirs(self, **kwargs):
        """
        Finds directories with data files.
        The names of the directories end with the scan number. Only the directories with a scan range and the ones covered by configuration are included.
        Parameters
        ----------
        prep_map : config object
            a configuration object containing experiment prep configuration parameters
        Returns
        -------
        dirs : list
            list of directories with raw data that will be included in prepared data
        scan_inxs : list
            list of scan numbers corresponding to the directories in the dirs list
        """
        no_scan_ranges = len(self.scan_ranges)
        unit_dirs_scan_indexes = {i : ([],[]) for i in range(no_scan_ranges)}

        def add_scan(scan_no, subdir):
            i = 0
            while scan_no > self.scan_ranges[i][1]:
                i += 1
                if i == no_scan_ranges:
                    return
            if scan_no >= self.scan_ranges[i][0]:
                # add the scan
                unit_dirs_scan_indexes[i][0].append(subdir)
                unit_dirs_scan_indexes[i][1].append(scan_no)

        try:
            data_dir = kwargs['data_dir'].replace(os.sep, '/')
        except:
            print('please provide data_dir in configuration file')
            return None, None

        def order_lists(dirs, inds):
            # The directory with the smallest index is placed as first, so all data files will
            # be alligned to the data file in this directory
            scans_order = np.argsort(inds).tolist()
            first_index = inds.pop(scans_order[0])
            first_dir = dirs.pop(scans_order[0])
            inds.insert(0, first_index)
            dirs.insert(0, first_dir)
            return dirs, inds

        for name in os.listdir(data_dir):
            subdir = data_dir + '/' + name
            if os.path.isdir(subdir):
                # exclude directories with fewer tif files than min_files
                if len(glob.glob1(subdir, "*.tif")) < self.min_files and len(glob.glob1(subdir, "*.tiff")) < self.min_files:
                    continue
                last_digits = re.search(r'\d+$', name)
                if last_digits is not None:
                    scan = int(last_digits.group())
                    if not scan in self.exclude_scans:
                        add_scan(scan, subdir)

        if self.separate_scan_ranges:
            for i in range(no_scan_ranges):
                if len(unit_dirs_scan_indexes[i]) > 1:
                    unit_dirs_scan_indexes[i] = order_lists(unit_dirs_scan_indexes[i][0], unit_dirs_scan_indexes[i][1])
        else:
            # combine all scans
            dirs = [unit_dirs_scan_indexes[i][0] for i in range(no_scan_ranges)]
            inds = [unit_dirs_scan_indexes[i][1] for i in range(no_scan_ranges)]
            unit_dirs_scan_indexes = (sum(dirs, []), sum(inds, []))
            unit_dirs_scan_indexes = order_lists(unit_dirs_scan_indexes[0], unit_dirs_scan_indexes[1])
        return unit_dirs_scan_indexes


    def prep_data(self, dirs_indexes, **kwargs):
        """
        Creates prep_data.tif file in <experiment_dir>/preprocessed_data directory or multiple prep_data.tif in <experiment_dir>/<scan_<scan_no>>/preprocessed_data directories.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        def combine_scans(refarr, dirs, nproc, scan=''):
            sumarr = np.zeros_like(refarr)
            sumarr = sumarr + refarr
            self.fft_refarr = np.fft.fftn(refarr)

            # https://www.edureka.co/community/1245/splitting-a-list-into-chunks-in-python
            # Need to further chunck becauase the result queue needs to hold N arrays.
            # if there are a lot of them and they are big, it runs out of ram.
            # since process takes 10-12 arrays, divide nscans/15 (to be safe) and make that many
            # chunks to pool.  Also ask between pools how much ram is avaiable and modify nproc.
            while (len(dirs) > 0):
                chunklist = dirs[0:min(len(dirs), nproc)]
                poollist = [dirs.pop(0) for i in range(len(chunklist))]
                with Pool(processes=nproc) as pool:
                    res = pool.map_async(self.read_align, poollist)
                    pool.close()
                    pool.join()
                for arr in res.get():
                    sumarr = sumarr + arr
            self.write_prep_arr(sumarr, scan)

        if self.separate_scan_ranges:
            single = []
            pops = []
            for key in dirs_indexes:
                (dirs, inds) = dirs_indexes[key]
                if len(dirs) == 0:
                    pops.append(key)
                elif len(dirs) == 1:
                    single.append((dirs_indexes[key][0][0], str(dirs_indexes[key][1][0])))
                    pops.append(key)
            for key in pops:
                dirs_indexes.pop(key)
            # process first the single scans
            if len(single) > 0:
                with Pool(processes=min(len(single), cpu_count())) as pool:
                    pool.starmap_async(self.read_write, single)
                    pool.close()
                    pool.join()
            # then process scan ranges
            if len(dirs_indexes) == 0:
                return
            pr = []
            for dir_ind in dirs_indexes.values():
                (dirs, inds) = dir_ind
                first_dir = dirs.pop(0)
                refarr = self.read_scan(first_dir)
                if refarr is None:
                    continue
                # estimate number of available cpus for each process
                arr_size = sys.getsizeof(refarr)
                nproc = int(ut.estimate_no_proc(arr_size, 15) / len(dirs_indexes))
                p = Process(target=combine_scans, args=(refarr, dirs, max(1, nproc), str(inds[0])+'-'+str(inds[-1])))
                p.start()
                pr.append(p)
            for p in pr:
                p.join()
        else:
            dirs, indexes = dirs_indexes[0], dirs_indexes[1]
            # dir_indexes consists of list of directories and corresponding list of indexes
            if len(dirs) == 1:
                arr = self.read_scan(dirs[0])
                if arr is not None:
                    self.write_prep_arr(arr)
                return

            if self.separate_scans:
                iterable = list(zip(dirs, [str(ix) for ix in indexes]))
                with Pool(processes=min(len(dirs_indexes), cpu_count())) as pool:
                    pool.starmap_async(self.read_write, iterable)
                    pool.close()
                    pool.join()
            else:
                first_dir = dirs.pop(0)
                refarr = self.read_scan(first_dir)
                if refarr is None:
                    return
                arr_size = sys.getsizeof(refarr)
                nproc = ut.estimate_no_proc(arr_size, 15)
                combine_scans(refarr, dirs, nproc)


    def read_scan(self, dir, **kwargs):
        """
        Reads raw data files from scan directory, applies correction, and returns 3D corrected data for a single scan directory.
        The correction is detector dependent. It can be darkfield and/ot whitefield correction.
        Parameters
        ----------
        dir : str
            directory to read the raw files from
        Returns
        -------
        arr : ndarray
            3D array containing corrected data for one scan.
        """
        files = []
        files_dir = {}
        for file in os.listdir(dir):
            if file.endswith('tif'):
                fnbase = file[:-4]
            elif file.endswith('tiff'):
                fnbase = file[:-4]
            else:
                continue
            last_digits = re.search(r'\d+$', fnbase)
            if last_digits is not None:
                key = int(last_digits.group())
                files_dir[key] = file

        ordered_keys = sorted(list(files_dir.keys()))

        for key in ordered_keys:
            file = files_dir[key]
            files.append(dir + '/' + file)

        # look at slice0 to find out shape
        n = 0
        try:
            slice0 = self.detector.get_frame(files[n], self.roi, self.Imult)
        except Exception as e:
            print(e)
            return None
        shape = (slice0.shape[0], slice0.shape[1], len(files))
        arr = np.zeros(shape, dtype=slice0.dtype)
        arr[:, :, 0] = slice0

        for file in files[1:]:
            n = n + 1
            slice = self.detector.get_frame(file, self.roi, self.Imult)
            arr[:, :, n] = slice
        return arr


    def write_prep_arr(self, arr, index='', **kwargs):
        """
        This clear the seam dependable on detector from the prepared array and saves the prepared data in <experiment_dir>/prep directory of
        experiment or <experiment_dir>/<scan_dir>/prep if writing for separate scans.
        """
        if index == '':
            prep_data_dir = self.experiment_dir + '/preprocessed_data'
        else:
            prep_data_dir = self.experiment_dir + '/scan_' + index + '/preprocessed_data'
        data_file = prep_data_dir + '/prep_data.tif'
        if not os.path.exists(prep_data_dir):
            os.makedirs(prep_data_dir)
        arr = self.detector.clear_seam(arr, self.roi)
        if not self.printed_dims:
            print('data array dimensions', arr.shape)
            self.printed_dims = True
        cohere.save_tif(arr, data_file)


    def get_detector_name(self):
        return self.det_name


    def set_detector(self, det_obj, prep_conf_map, **kwargs):
        # The detector attributes for background/whitefield/etc need to be set to read frames
        self.detector = det_obj

        # if anything in config file has the same name as a required detector attribute, copy it to
        # the detector
        # this will capture things like whitefield_filename, etc.
        for attr in prep_conf_map.keys():
            if hasattr(self.detector, attr):
                setattr(self.detector, attr, prep_conf_map.get(attr))


    def read_write(self, scan_dir, index, **kwargs):
        arr = self.read_scan(scan_dir)
        self.write_prep_arr(arr, index)


    def read_align(self, dir, **kwargs):
        """
        Aligns scan with reference array.  Referrence array is field of this class.
        Parameters
        ----------
        dir : str
            directory to the raw data
        Returns
        -------
        aligned_array : array
            aligned array
        """
        # read
        arr = self.read_scan(dir)
        # align
        return np.abs(ut.shift_to_ref_array(self.fft_refarr, arr))
