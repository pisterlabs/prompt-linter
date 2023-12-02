import os, time
import multiprocessing
import logging
import pickle
import shutil
import click
import pycwb
import matplotlib.pyplot as plt

from pycwb.modules.plot.waveform import plot_reconstructed_waveforms
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.utils.dep_check import check_dependencies

if check_dependencies(['autoencoder', 'reconstruction', 'logger', 'read_data', 'data_conditioning', 'coherence',
                       'super_cluster', 'likelihood', 'job_segment', 'catalog', 'plot', 'plot_map', 'web_viewer']):
    exit(1)

from pycwb.config import Config
from pycwb.types.network import Network
from pycwb.modules.autoencoder import get_glitchness
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.logger import logger_init
from pycwb.modules.read_data import read_from_job_segment, generate_injection
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.coherence import coherence
from pycwb.modules.super_cluster import supercluster
from pycwb.modules.likelihood import likelihood
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.catalog import create_catalog, add_events_to_catalog
from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.web_viewer.create import create_web_viewer
from pycwb.modules.plot_map.world_map import plot_world_map, plot_skymap_contour

logger = logging.getLogger(__name__)


def analyze_job_segment(config, job_seg, plot, compress_json):
    """Analyze one job segment with the given configuration

    This function includes the following stages:

    1. Read data from job segment (pycwb.modules.read_data.read_from_job_segment) \n
    2. Data conditioning (pycwb.modules.data_conditioning.data_conditioning) \n
    3. Create network (pycwb.modules.coherence.create_network) \n
    4. Coherence (pycwb.modules.coherence.coherence) \n
    5. Supercluster (pycwb.modules.super_cluster.supercluster) \n
    6. Likelihood (pycwb.modules.likelihood.likelihood) \n

    The results will be saved to the output directory in json format on likelihood stage

    :param config: configuration
    :type config: Config
    :param job_seg: job segment
    :type job_seg: WaveSegment
    """
    # config, job_seg = args
    start_time = time.perf_counter()

    job_id = job_seg.index
    # log job info
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Start time: {job_seg.start_time}")
    logger.info(f"End time: {job_seg.end_time}")
    logger.info(f"Duration: {job_seg.end_time - job_seg.start_time}")

    # read data
    data = None
    if job_seg.frames:
        data = read_from_job_segment(config, job_seg)
    if job_seg.injections:
        data = generate_injection(config, job_seg, data)

    # data conditioning
    tf_maps, nRMS_list = data_conditioning(config, data)

    # calculate coherence
    # TODO: Merge resolution here?
    fragment_clusters = coherence(config, tf_maps, nRMS_list)

    # create network
    network = Network(config, tf_maps, nRMS_list)

    # supercluster
    pwc_list = supercluster(config, network, fragment_clusters, tf_maps)

    # likelihood
    events, clusters, skymap_statistics = likelihood(config, network, pwc_list)

    with multiprocessing.Pool(processes=min(config.nproc, len(events))) as pool:
        pool.starmap(post_production, [(config, job_id, event, cluster, event_skymap_statistics, plot, compress_json)
                                       for event, cluster, event_skymap_statistics in zip(events, clusters, skymap_statistics)])
    # for i, tf_map in enumerate(tf_maps):
    #     plot_event_on_spectrogram(tf_map, events, filename=f'{config.outputDir}/events_{job_id}_all_{i}.png')

    # calculate the performance
    end_time = time.perf_counter()
    logger.info("-" * 80)
    logger.info(f"Job {job_id} finished in {round(end_time - start_time, 1)} seconds")
    logger.info(f"Speed factor: {round((job_seg.end_time - job_seg.start_time) / (end_time - start_time), 1)}X")
    logger.info("-" * 80)


def post_production(config, job_id, event, cluster, event_skymap_statistics, plot, compress_json):
    # extra info will be saved
    extra_info = {}

    # create event folder
    trigger_folder = f"{config.outputDir}/trigger_{job_id}_{event.stop[0]}_{event.hash_id}"
    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)

    # save the results
    save_dataclass_to_json(event, f'{trigger_folder}/event.json', compress_json=compress_json)
    save_dataclass_to_json(cluster, f'{trigger_folder}/cluster.json', compress_json=compress_json)
    # save the skymap statistics as json file
    save_dataclass_to_json(event_skymap_statistics, f'{trigger_folder}/skymap_statistics.json', compress_json=compress_json)
    # save event to catalog
    add_events_to_catalog(f"{config.outputDir}/catalog.json", event.summary(job_id, f"{event.stop[0]}_{event.hash_id}"))

    # post-production only for selected events
    if cluster.cluster_status != -1:
        return

    reconstructed_waves = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                               'signal', 0, True)

    # calculate the glitchness
    glitchness = get_glitchness(config, reconstructed_waves, event.sSNR, event.likelihood)
    # TODO: save to event file
    print(f"Glitchness: {glitchness}")
    extra_info['glitchness'] = glitchness[0][0]

    # save the extra info
    save_dataclass_to_json(extra_info, f'{trigger_folder}/extra_info.json', compress_json=compress_json)

    if plot:
        plot_reconstructed_waveforms(trigger_folder, reconstructed_waves,
                                     xlim=(event.left[0], event.left[0] + event.stop[0] - event.start[0]))
        # plot the likelihood map
        plot_statistics(cluster, 'likelihood', filename=f'{trigger_folder}/likelihood_map.png')
        plot_statistics(cluster, 'null', filename=f'{trigger_folder}/null_map.png')

        # plot_world_map(event.phi[0], event.theta[0], filename=f'{config.outputDir}/world_map_{job_id}_{i+1}.png')
        for key in event_skymap_statistics.keys():
            plot_skymap_contour(event_skymap_statistics,
                                key=key,
                                reconstructed_loc=(event.phi[0], event.theta[0]),
                                detector_loc=(event.phi[3], event.theta[3]),
                                resolution=1,
                                filename=f'{trigger_folder}/{key}.png')


def search(user_parameters='./user_parameters.yaml', working_dir=".", log_file=None, log_level='INFO',
           no_subprocess=False, overwrite=False, nproc=None, plot=True, compress_json=True):
    """Main function to run the search

    This function will read the user parameters, select the job segments, create the catalog,
    copy the html and css files and run the search in subprocesses by default to avoid memory leak.

    Parameters
    ----------
    user_parameters : str, optional
        path to user parameters file, by default './user_parameters.yaml'
    working_dir : str, optional
        working directory, by default "."
    log_file : str, optional
        path to log file, by default None
    log_level : str, optional
        log level, by default 'INFO'
    no_subprocess : bool, optional
        run the search in the main process, by default False (Set to True for macOS development)
    overwrite : bool, optional
        overwrite the existing results, by default False
    nproc : int, optional
        number of threads to use, by default None (use the value in user parameters)
    plot : bool, optional
        plot the results, by default True
    compress_json : bool, optional
        compress the json files, by default True
    """
    # create working directory
    working_dir = os.path.abspath(working_dir)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    os.chdir(working_dir)

    # initialize logger
    logger_init(log_file, log_level)
    logger.info(f"Working directory: {os.path.abspath(working_dir)}")

    # set env HOME_WAT_FILTERS
    if not os.environ.get('HOME_WAT_FILTERS'):
        logger.warning("HOME_WAT_FILTERS is not set, default to pycwb/vendor")
        logger.warning("Please download the latest version of cwb config and set HOME_WAT_FILTERS to the path of folder XTALKS")
        pycwb_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        os.environ['HOME_WAT_FILTERS'] = f"{os.path.abspath(pycwb_path)}/vendor"
        logger.info(f"Set HOME_WAT_FILTERS to {os.environ['HOME_WAT_FILTERS']}")

    # read config
    logger.info("Reading user parameters")
    config = Config(user_parameters)

    # overwrite threads if it is set
    if nproc:
        config.nproc = nproc

    # Safety Check: if output is not empty, ask for confirmation
    if os.path.exists(config.outputDir) and os.listdir(config.outputDir):
        if overwrite:
            logger.info(f"Overwrite output directory {config.outputDir}")
        elif not click.confirm(f"Output directory {config.outputDir} is not empty, do you want to continue?"):
            logger.info("Search stopped")
            return

    # create folder for output and log
    logger.info(f"Output folder: {working_dir}/{config.outputDir}")
    logger.info(f"Log folder: {working_dir}/{config.logDir}")
    if not os.path.exists(config.outputDir):
        os.makedirs(config.outputDir)
    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)

    # copy user parameters to output folder if it is not there
    if not os.path.exists(f"{config.outputDir}/user_parameters.yaml"):
        shutil.copyfile(user_parameters, f"{config.outputDir}/user_parameters.yaml")
    else:
        logger.warning(f"User parameters file already exists in {working_dir}/{config.outputDir}")

    # select job segments
    job_segments = create_job_segment_from_config(config)

    # create catalog
    logger.info("Creating catalog file")
    create_catalog(f"{config.outputDir}/catalog.json", config, job_segments)

    # copy all files in web_viewer to output folder
    create_web_viewer(config.outputDir)

    # analyze job segments
    logger.info("Start analyzing job segments")
    for job_seg in job_segments:
        if no_subprocess:
            analyze_job_segment(config, job_seg, plot=plot, compress_json=compress_json)
            # gc.collect()
        else:
            process = multiprocessing.Process(target=analyze_job_segment, args=(config, job_seg, plot, compress_json))
            process.start()
            process.join()
