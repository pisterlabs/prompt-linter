#!/usr/bin/env python3 
"""
Time-series PGE wrapper.
"""
from __future__ import division

from builtins import str
from past.utils import old_div
import os, sys, re, requests, json, shutil, traceback, logging, pickle
import argparse, multiprocessing, hashlib, h5py
from datetime import datetime
from subprocess import check_call
from glob import glob
import numpy as np
from osgeo import gdal
from gdalconst import GA_ReadOnly

import matplotlib
matplotlib.use("Agg")

import isce
from iscesys.Component.ProductManager import ProductManager as PM

from utils.UrlUtils import UrlUtils

import ts_common

gdal.UseExceptions() # make GDAL raise python exceptions


log_format = "[%(asctime)s: %(levelname)s/%(funcName)s] %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger('create_ts_roi')


BASE_PATH = os.path.dirname(__file__)


DT_RE = re.compile(r'^(\d{4})-(\d{2})-(\d{2})')
TN_RE = re.compile(r'_TN(\d+)_')
S1_RE = re.compile(r'^S1\w$')


ID_TMPL = "time-series_{project}-{startdt}Z-{enddt}Z-{hash}-{version}"


# read in example.rsc template
with open(os.path.join(BASE_PATH, "example.rsc.tmpl")) as f:
    RSC_TMPL = f.read()


# read in prepdataxml.py template
with open(os.path.join(BASE_PATH, "prepdataxml.py.tmpl")) as f:
    PREPDATA_TMPL = f.read()


# read in prepsbasxml.py template
with open(os.path.join(BASE_PATH, "prepsbasxml.py.tmpl")) as f:
    PREPSBAS_TMPL = f.read()


def check_ts(es_url, es_index, id):
    """Query for time-series with specified input ID."""

    query = {
        "query":{
            "bool":{
                "must":[
                    {"term":{"id":id}},
                ]
            }
        },
        "fields": [],
    }

    if es_url.endswith('/'):
        search_url = '%s%s/_search' % (es_url, es_index)
    else:
        search_url = '%s/%s/_search' % (es_url, es_index)
    r = requests.post(search_url, data=json.dumps(query))
    if r.status_code != 200:
        logger.info("Failed to query {}:\n{}".format(es_url, r.text))
        logger.info("query: {}".format(json.dumps(query, indent=2)))
        logger.info("returned: {}".format(r.text))
    r.raise_for_status()
    result = r.json()
    logger.info('dedup check: {}'.format(json.dumps(result, indent=2)))
    total = result['hits']['total']
    if total == 0: id = 'NONE'
    else: id = result['hits']['hits'][0]['_id']
    return total, id


def ts_exists(es_url, es_index, id):
    """Check time-series exists in GRQ."""

    total, id = check_ts(es_url, es_index, id)
    if total > 0: return True
    return False


#def download_file(url, outdir='.', session=None):
#    """Download file to specified directory."""
#
#    if session is None: session = requests.session()
#    path = os.path.join(outdir, os.path.basename(url))
#    logger.info('Downloading URL: {}'.format(url))
#    r = session.get(url, stream=True, verify=False)
#    try:
#        val = r.raise_for_status()
#        success = True
#    except:
#        success = False
#    if success:
#        with open(path,'wb') as f:
#            for chunk in r.iter_content(chunk_size=1024):
#                if chunk:
#                    f.write(chunk)
#                    f.flush()
#    return success


def call_noerr(cmd):
    """Run command and warn if exit status is not 0."""

    try: check_call(cmd, shell=True)
    except Exception as e:
        logger.warn("Got exception running {}: {}".format(cmd, str(e)))
        logger.warn("Traceback: {}".format(traceback.format_exc()))


def gdal_translate(vrt_in, vrt_out, min_lat, max_lat, min_lon, max_lon, no_data, band):
    """Run gdal_translate to project image to a region of interest bbox."""

    cmd_tmpl = "gdal_translate -of VRT -a_nodata {} -projwin {} {} {} {} -b {} {} {}"
    return check_call(cmd_tmpl.format(no_data, min_lon, max_lat, max_lon,
                                      min_lat, band, vrt_in, vrt_out), shell=True)


def main(input_json_file):
    """HySDS PGE wrapper for time-series generation."""

    # save cwd (working directory)
    cwd = os.getcwd()

    # get time-series input
    input_json_file = os.path.abspath(input_json_file)
    if not os.path.exists(input_json_file):
        raise RuntimeError("Failed to find %s." % input_json_file)
    with open(input_json_file) as f:
        input_json = json.load(f)
    logger.info("input_json: {}".format(json.dumps(input_json, indent=2)))

    # get coverage threshold
    covth = input_json['coverage_threshold']

    # get coherence threshold
    cohth = input_json['coherence_threshold']

    # get range and azimuth pixel size
    range_pixel_size = input_json['range_pixel_size']
    azimuth_pixel_size = input_json['azimuth_pixel_size']

    # get incidence angle
    inc = input_json['inc']

    # get filt
    filt = input_json['filt']

    # network and gps deramp
    netramp = input_json['netramp']
    gpsramp = input_json['gpsramp']

    # get region of interest
    if input_json['region_of_interest']:
        logger.info("Running Time Series with Region of Interest")
        min_lat, max_lat, min_lon, max_lon = input_json['region_of_interest']
    else:
        logger.info("Running Time Series on full data")
        min_lon, max_lon, min_lat, max_lat = ts_common.get_envelope(input_json['products'])
        logger.info("env: {} {} {} {}".format(min_lon, max_lon, min_lat, max_lat))

    # get reference point in radar coordinates and length/width for box
    ref_lat, ref_lon = input_json['ref_point']
    ref_width = int(old_div((input_json['ref_box_num_pixels'][0]-1),2))
    ref_height = int(old_div((input_json['ref_box_num_pixels'][1]-1),2))

    # align images
    center_lines_utc = []
    ifg_info = {}
    ifg_coverage = {}
    for prod_num, ifg_prod in enumerate(input_json['products']):
        logger.info('#' * 80)
        logger.info('Processing: {} ({} of {}) (current stack count: {})'.format(
                    ifg_prod, prod_num+1, len(input_json['products'])+1, len(ifg_info)))
        logger.info('-' * 80)

        # get IFG metadata
        ifg_met_file = glob("{}/*.met.json".format(ifg_prod))[0]
        with open(ifg_met_file) as f:
            ifg_met = json.load(f)

        # filter out product from different subswath
        swath = ifg_met['swath'][0] if isinstance(ifg_met['swath'], list) else ifg_met['swath']
        if swath != input_json['subswath']:
            logger.info('Filtered out {}: unmatched subswath {}'.format(ifg_prod,
                        ifg_met['swath']))
            continue

        # extract sensing start and stop dates
        match = DT_RE.search(ifg_met['sensingStart'])
        if not match: raise RuntimeError("Failed to extract start date.")
        start_dt = ''.join(match.groups())
        match = DT_RE.search(ifg_met['sensingStop'])
        if not match: raise RuntimeError("Failed to extract stop date.")
        stop_dt = ''.join(match.groups())
        logger.info('start_dt: {}'.format(start_dt))
        logger.info('stop_dt: {}'.format(stop_dt))

        # extract perpendicular baseline and sensor for ifg.list input file
        cb_pkl = os.path.join(ifg_prod, "PICKLE", "computeBaselines")
        with open(cb_pkl, 'rb') as f:
            catalog = pickle.load(f)
        bperp = ts_common.get_bperp(catalog)
        sensor = catalog['master']['sensor']['mission']
        if sensor is None: sensor = catalog['slave']['sensor']['mission']
        if sensor is None and catalog['master']['sensor']['imagingmode'] == "TOPS":
            sensor = "S1X"
        if sensor is None:
            logger.warn("{} will be thrown out. Failed to extract sensor".format(ifg_prod))
            continue

        # set no data value
        if S1_RE.search(sensor):
            sensor = "S1"
            no_data = 0.
        elif sensor == "SMAP": no_data = -9999.
        else:
            raise RuntimeError("Unknown sensor: {}".format(sensor))

        # project unwrapped phase and correlation products to common region_of_interest bbox (ROI)
        unw_vrt_in = os.path.join(ifg_prod, "merged", "filt_topophase.unw.geo.vrt")
        unw_vrt_out = os.path.join(ifg_prod, "merged", "aligned.unw.vrt")
        gdal_translate(unw_vrt_in, unw_vrt_out, min_lat, max_lat, min_lon, max_lon, no_data, 2)
        cor_vrt_in = os.path.join(ifg_prod, "merged", "phsig.cor.geo.vrt")
        cor_vrt_out = os.path.join(ifg_prod, "merged", "aligned.cor.vrt")
        gdal_translate(cor_vrt_in, cor_vrt_out, min_lat, max_lat, min_lon, max_lon, no_data, 1)

        # get width and length of aligned/projected images and
        # determine reference point limits
        ds = gdal.Open(cor_vrt_out, GA_ReadOnly)
        gt = ds.GetGeoTransform()
        width = ds.RasterXSize
        length = ds.RasterYSize
        ref_line  = int(old_div((ref_lat - gt[3]), gt[5]))
        ref_pixel = int(old_div((ref_lon - gt[0]), gt[1]))
        xlim = [0, width]
        ylim = [0, length]
        rxlim = [ref_pixel - ref_width, ref_pixel + ref_width]
        rylim = [ref_line - ref_height, ref_line + ref_height]
        #logger.info("rxlim: {}".format(rxlim))
        #logger.info("rylim: {}".format(rylim))

        # read the coherence data and build mask from coherence threshold
        band = ds.GetRasterBand(1)
        cor = band.ReadAsArray()
        cor_ref = cor[rylim[0]:rylim[1], rxlim[0]:rxlim[1]]
        logger.info("cor_ref: {} {}".format(cor_ref.shape, cor_ref))
        ds = None
        #logger.info("cor: {} {}".format(cor.shape, cor))
        mask = np.nan*np.ones(cor.shape)
        mask[cor >= cohth] = 1.0
        #logger.info("mask_ref: {} {}".format(mask_ref.shape, mask_ref))

        # read the phase data and mask out reference bbox pixels with no data
        ds = gdal.Open(unw_vrt_out, GA_ReadOnly)
        band = ds.GetRasterBand(1)
        phs = band.ReadAsArray()
        ds = None
        #logger.info("phs: {} {}".format(phs.shape, phs))
        mask[phs == no_data] = np.nan
        phs_ref = phs[rylim[0]:rylim[1], rxlim[0]:rxlim[1]]
        mask_ref = mask[rylim[0]:rylim[1], rxlim[0]:rxlim[1]]
        phs_ref = phs_ref*mask_ref
        #logger.info("phs_ref: {} {}".format(phs_ref.shape, phs_ref))
        phs_ref_mean = np.nanmean(phs_ref)
        logger.info("phs_ref mean: {}".format(phs_ref_mean))

        # filter out product with no valid phase data in reference bbox
        # or did not pass coherence threshold
        if np.isnan(phs_ref_mean):
            logger.info('Filtered out {}: no valid data in ref bbox'.format(ifg_prod))
            continue

        # filter out product with ROI coverage of valid data less than threshold
        #cov = np.sum(~np.isnan(mask))/(mask.size*1.)
        #logger.info('coverage: {}'.format(cov))
        #if cov < covth:
        #    logger.info('Filtered out {}: ROI coverage of valid data was below threshold ({} vs. {})'.format(
        #                ifg_prod, cov, covth))
        #    continue

        # filter out product with ROI latitude coverage of valid data less than threshold
        cov = old_div(np.sum(~np.isnan(mask), axis=0).max(),(mask.shape[0]*1.))
        logger.info('coverage: {}'.format(cov))
        if cov < covth:
            logger.info('Filtered out {}: ROI latitude coverage of valid data was below threshold ({} vs. {})'.format(
                        ifg_prod, cov, covth))
            continue

        # get wavelength, heading degree and center line UTC
        ifg_xml = os.path.join(ifg_prod, "fine_interferogram.xml")
        pm = PM()
        pm.configure()
        ifg_obj = pm.loadProduct(ifg_xml)
        wavelength = ifg_obj.bursts[0].radarWavelength
        sensing_mid = ifg_obj.bursts[0].sensingMid
        heading_deg = ifg_obj.bursts[0].orbit.getENUHeading(sensing_mid)
        center_line_utc = int((sensing_mid - datetime(year=sensing_mid.year,
                                                      month=sensing_mid.month,
                                                      day=sensing_mid.day)).total_seconds())
        # track sensing mid
        center_lines_utc.append(sensing_mid)

        # create date ID
        dt_id = "{}_{}".format(start_dt, stop_dt)

        # use IFG product with larger coverage
        if os.path.exists(dt_id):
            if cov <= ifg_coverage[dt_id]:
                logger.info('Filtered out {}: already exists with larger coverage ({} vs. {})'.format(
                            ifg_prod, ifg_coverage[dt_id], cov))
                continue
            else:
                logger.info('Larger coverage found for {} ({} vs. {})'.format(
                            dt_id, cov, ifg_coverage[dt_id]))
                os.unlink(dt_id)

        # create soft link for aligned products
        os.symlink(ifg_prod, dt_id)

        # set ifg list info
        ifg_info[dt_id] = {
            'product': ifg_prod,
            'start_dt': start_dt,
            'stop_dt': stop_dt,
            'bperp': bperp,
            'sensor': sensor,
            'width': width,
            'length': length,
            'xlim': xlim,
            'ylim': ylim,
            'rxlim': rxlim,
            'rylim': rylim,
            'cohth': cohth,
            'wavelength': wavelength,
            'heading_deg': heading_deg,
            'center_line_utc': center_line_utc,
            'range_pixel_size': range_pixel_size,
            'azimuth_pixel_size': azimuth_pixel_size,
            'inc': inc,
            'netramp': netramp,
            'gpsramp': gpsramp,
            'filt': filt,
            'unw_vrt_in': unw_vrt_in,
            'unw_vrt_out': unw_vrt_out,
            'cor_vrt_in': cor_vrt_in,
            'cor_vrt_out': cor_vrt_out,
        }

        # track coverage
        ifg_coverage[dt_id] = cov

        # log success status
        logger.info('Added {} to final input stack'.format(ifg_prod))

    # print status after filtering
    logger.info("After filtering: {} out of {} will be used for GIAnT processing".format(
                len(ifg_info), len(input_json['products'])))

    # croak no products passed filters
    if len(ifg_info) == 0:
        raise RuntimeError("All products in the stack were filtered out. Check thresholds.")

    # get sorted ifg date list
    ifg_list = sorted(ifg_info)

    # get endpoint configurations
    uu = UrlUtils()
    es_url = uu.rest_url
    es_index = uu.grq_index_prefix
    logger.info("GRQ url: {}".format(es_url))
    logger.info("GRQ index: {}".format(es_index))

    # get hash of all params
    m = hashlib.new('md5')
    m.update("{} {} {} {}".format(min_lon, max_lon, min_lat, max_lat).encode('utf-8'))
    m.update("{} {}".format(*input_json['ref_point']).encode('utf-8'))
    m.update("{} {}".format(*input_json['ref_box_num_pixels']).encode('utf-8'))
    m.update("{}".format(cohth).encode('utf-8'))
    m.update("{}".format(range_pixel_size).encode('utf-8'))
    m.update("{}".format(azimuth_pixel_size).encode('utf-8'))
    m.update("{}".format(inc).encode('utf-8'))
    m.update("{}".format(netramp).encode('utf-8'))
    m.update("{}".format(gpsramp).encode('utf-8'))
    m.update("{}".format(filt).encode('utf-8'))
    m.update(" ".join(ifg_list).encode('utf-8'))
    roi_ref_hash = m.hexdigest()[0:5]

    # get time series product ID
    center_lines_utc.sort()
    id = ID_TMPL.format(project=input_json['project'].replace(' ', '_'),
                        startdt=center_lines_utc[0].strftime('%Y%m%dT%H%M%S'),
                        enddt=center_lines_utc[-1].strftime('%Y%m%dT%H%M%S'),
                        hash=roi_ref_hash, version=uu.version)
    logger.info("Product ID for version {}: {}".format(uu.version, id))

    # check if time-series already exists
    if ts_exists(es_url, es_index, id):
        logger.info("{} time-series for {}".format(uu.version, id) +
                    " was previously generated and exists in GRQ database.")

    # write ifg.list
    with open ('ifg.list', 'w') as f:
        for i, dt_id in enumerate(ifg_list):
            logger.info("{start_dt} {stop_dt} {bperp:7.2f} {sensor} {width} {length} {wavelength} {heading_deg} {center_line_utc} {xlim} {ylim} {rxlim} {rylim}\n".format(**ifg_info[dt_id]))
            f.write("{start_dt} {stop_dt} {bperp:7.2f} {sensor}\n".format(**ifg_info[dt_id]))

            # write input files on first ifg
            if i == 0:
                # write example.rsc
                with open('example.rsc', 'w') as g:
                    g.write(RSC_TMPL.format(**ifg_info[dt_id]))

                # write prepdataxml.py
                with open('prepdataxml.py', 'w') as g:
                    g.write(PREPDATA_TMPL.format(**ifg_info[dt_id]))

                # write prepsbasxml.py
                with open('prepsbasxml.py', 'w') as g:
                    g.write(PREPSBAS_TMPL.format(nvalid=len(ifg_list), **ifg_info[dt_id]))

    # copy userfn.py
    shutil.copy(os.path.join(BASE_PATH, "userfn.py"), "userfn.py")

    # get aligned coherence file for adding geocoding info to products
    cor_vrt = ifg_info[ifg_list[0]]['cor_vrt_out']

    # create data.xml
    logger.info("Running step 1: prepdataxml.py")
    check_call("python prepdataxml.py", shell=True)

    # prepare interferogram stack
    logger.info("Running step 2: PrepIgramStack.py")
    check_call("{}/PrepIgramStackWrapper.py".format(BASE_PATH), shell=True)

    # create sbas.xml
    logger.info("Running step 3: prepsbasxml.py")
    check_call("python prepsbasxml.py", shell=True)

    # stack preprocessing: apply atmospheric corrections and estimate residual orbit errors
    logger.info("Running step 4: ProcessStack.py")
    check_call("{}/ProcessStackWrapper.py".format(BASE_PATH), shell=True)

    # SBASInvert.py to create time-series using short baseline approach (least-squares)
    logger.info("Running step 5: SBASInvert.py")
    check_call("{}/SBASInvertWrapper.py".format(BASE_PATH), shell=True)

    # add lat, lon, and time datasets to LS-PARAMS.h5 for THREDDS
    sbas = os.path.join("Stack", "LS-PARAMS.h5")
    check_call("{}/prep_tds.py {} {}".format(BASE_PATH, cor_vrt, sbas), shell=True)

    # NSBASInvert.py to create time-series using partially coherent pixels approach
    logger.info("Running step 6: NSBASInvert.py")
    cpu_count = multiprocessing.cpu_count()
    check_call("{}/NSBASInvertWrapper.py -nproc {}".format(BASE_PATH, cpu_count), shell=True)

    # add lat, lon, and time datasets to NSBAS-PARAMS.h5 for THREDDS
    nsbas = os.path.join("Stack", "NSBAS-PARAMS.h5")
    check_call("{}/prep_tds.py {} {}".format(BASE_PATH, cor_vrt, nsbas), shell=True)

    # SBASxval.py determine stats to estimate uncertainties (leave-one-out approach)
    #logger.info("Running step 7: SBASxval.py")
    #check_call("{}/SBASxvalWrapper.py".format(BASE_PATH), shell=True)

    # add lat, lon, and time datasets to LS-xval.h5 for THREDDS
    #xval = os.path.join("Stack", "LS-xval.h5")
    #check_call("{}/prep_tds.py {} {}".format(BASE_PATH, cor_vrt, xval), shell=True)

    # extract timestep dates
    h5f = h5py.File(nsbas, 'r')
    times = h5f.get('time')[:]
    h5f.close()
    timesteps = [datetime.fromtimestamp(i).isoformat('T') for i in times[:]]

    # create product directory
    prod_dir = id
    os.makedirs(prod_dir, 0o755)
    #Compute bounding polygon before 
    bound_polygon = None
    try:
        bound_polygon = ts_common.get_bounding_polygon("./Slack/NSBAS-PARAMS.h5")
    except Exception as e:
        logger.warn("Using less precise BBOX due to error. {0}.{1}".format(type(e),e))
    # move and compress HDF5 products
    prod_files = glob("Stack/*")
    for i in prod_files:
        shutil.move(i, prod_dir)
        check_call("pigz -f -9 {}".format(os.path.join(prod_dir, os.path.basename(i))), shell=True)

    # create browse image
    png_files = glob("Figs/Igrams/*.png")
    shutil.copyfile(png_files[0], os.path.join(prod_dir, "browse.png"))
    call_noerr("convert -resize 250x250 {} {}".format(png_files[0],
               os.path.join(prod_dir, "browse_small.png")))

    # copy pngs
    for i in png_files: shutil.move(i, prod_dir)

    # save other files to product directory
    shutil.copyfile(input_json_file, os.path.join(prod_dir,"{}.context.json".format(id)))
    shutil.copyfile("data.xml", os.path.join(prod_dir, "data.xml"))
    shutil.copyfile("example.rsc", os.path.join(prod_dir, "example.rsc"))
    shutil.copyfile("ifg.list", os.path.join(prod_dir, "ifg.list"))
    shutil.copyfile("prepdataxml.py", os.path.join(prod_dir, "prepdataxml.py"))
    shutil.copyfile("prepsbasxml.py", os.path.join(prod_dir, "prepsbasxml.py"))
    shutil.copyfile("sbas.xml", os.path.join(prod_dir, "sbas.xml"))

    # create met json
    met = {
        "bbox": [
          [ max_lat, max_lon ],
          [ max_lat, min_lon ],
          [ min_lat, min_lon ],
          [ min_lat, max_lon ],
        ], 
        "dataset_type": "time-series", 
        "product_type": "time-series", 
        "reference": False, 
        "sensing_time_initial": timesteps[0],
        "sensing_time_final": timesteps[-1],
        "sensor": "SAR-C Sentinel1",
        "tags": [ input_json['project'] ],
        "trackNumber": int(TN_RE.search(input_json['products'][0]).group(1)),
        "swath": input_json['subswath'],
        "ifg_count": len(ifg_info),
        "ifgs": [ifg_info[i]['product'] for i in sorted(ifg_info)],
        "timestep_count": len(timesteps),
        "timesteps": timesteps,
      }
    #Set a better bbox
    if not bound_polygon is None:
        met["bbox"] = bound_polygon
    met_file = os.path.join(prod_dir, "{}.met.json".format(id))
    with open(met_file, 'w') as f:
        json.dump(met, f, indent=2)
    #Dataset JSON
    ts_common.write_dataset_json(prod_dir,id,met["bbox"],timesteps[0],timesteps[-1]) 
    # write PROV-ES JSON
    
    # clean out SAFE directories and symlinks
    for i in input_json['products']: shutil.rmtree(i)
    for i in ifg_list: os.unlink(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json_file", help="input JSON file")
    args = parser.parse_args()
    try: main(args.input_json_file)
    except Exception as e:
        with open('_alt_error.txt', 'w') as f:
            f.write("%s\n" % str(e))
        with open('_alt_traceback.txt', 'w') as f:
            f.write("%s\n" % traceback.format_exc())
        raise
    sys.exit(0)
