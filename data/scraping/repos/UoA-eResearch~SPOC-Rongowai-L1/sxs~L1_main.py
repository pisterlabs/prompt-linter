# mike.laverick@auckland.ac.nz
# L1_main_L0.py
import argparse
import os
from pathlib import Path
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)
from aeff import aeff_and_nbrcs
from brcs import brcs_calculations
from calibration import ddm_calibration
from fresnel import fresnel_calculations
from gps import calculate_satellite_orbits
from load_files import L0_file, input_files
from noise import noise_floor_prep, noise_floor, confidence_flag
from quality_flags import quality_flag_calculations
from specular import specular_calculations
from coherence import coherence_detection
from output import L1_file, write_netcdf
from utils import OrbitFileDelayError


def process_L1s(L0_filename, L1_filename, inp, L1_DICT, settings):
    # Prelaunch 1: Load L0 data, filter valid timestamps, and smooth out 0 values
    L0 = L0_file(L0_filename)

    # Part 1: General processing
    # This part derives global constants, timestamps, and all the other
    # parameters at ddm timestamps
    L1 = L1_file(L1_filename, settings, L0, inp)

    # Part 2: Derive TX related variables
    # This part derives TX positions and velocities, maps between PRN and SVN,
    # and gets track ID
    calculate_satellite_orbits(settings, L0, L1, inp)

    # Part 3A: SP solver and geometries
    specular_calculations(
        L0,
        L1,
        inp,
        L1.rx_pos_x,
        L1.rx_pos_y,
        L1.rx_pos_z,
        L1.rx_vel_x,
        L1.rx_vel_y,
        L1.rx_vel_z,
        L1.rx_roll,
        L1.rx_pitch,
    )

    # Part 3B: noise floor, SNR
    noise_floor_prep(
        L0,
        L1,
        L1.postCal["add_range_to_sp"],
        L1.rx_pos_x,
        L1.rx_pos_y,
        L1.rx_pos_z,
        L1.rx_vel_x,
        L1.rx_vel_y,
        L1.rx_vel_z,
    )
    noise_floor(L0, L1)

    # Part 4: L1a calibration
    # this part converts from raw counts to signal power in watts and complete
    # L1a calibration
    ddm_calibration(
        inp,
        L0,
        L1
        # L0.std_dev_rf1,
        # L0.std_dev_rf2,
        # L0.std_dev_rf3,
        # L0.J,
        # L1.postCal["prn_code"],
        # L0.raw_counts,
        # L0.rf_source,
        # L0.first_scale_factor,
        # L1.ddm_power_counts,
        # L1.power_analog,
        # L1.postCal["ddm_ant"],
        # L1.postCal["inst_gain"],
        # L1.postCal["ddm_noise_floor"]
    )

    # Part 3C: confidence flag of the SP solved
    confidence_flag(L0, L1)

    # Part 5: Copol and xpol BRCS, reflectivity, peak reflectivity
    brcs_calculations(L0, L1)

    # Part 6: NBRCS and other related parameters
    aeff_and_nbrcs(L0, L1, inp, L1.rx_vel_x, L1.rx_vel_y, L1.rx_vel_z, L1.rx_pos_lla)

    # Part 7: coherence detection
    coherence_detection(L0, L1, L1.rx_pos_lla)

    # Part 8: fresnel dimensions and cross Pol
    fresnel_calculations(L0, L1)

    # Quality Flags
    quality_flag_calculations(
        L0,
        L1,
        L1.rx_roll,
        L1.rx_pitch,
        L1.rx_heading,
        L1.postCal["ant_temp_nadir"],
        L1.postCal["add_range_to_sp"],
        L1.rx_pos_lla,
        L1.rx_vel_x,
        L1.rx_vel_y,
        L1.rx_vel_z,
    )

    # write to netcdf
    L1.add_to_postcal(L0)
    write_netcdf(L1.postCal, L1_DICT, L1.filename)
    # clear processed L0 and L1 files/variables from memory
    L0 = None
    L1 = None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Process L1 science file from L0 netCDF."
    )
    argparser.add_argument(
        "--conf-file",
        type=str,
        help="path for configuration file that specifies I/O and version settings. Default='../config.sh'",
    )
    args = argparser.parse_args()

    argparser.add_argument(
        "--input-L0-dir",
        type=str,
        help="User-specified L0 input file directory to override conf file.",
    )
    args = argparser.parse_args()

    argparser.add_argument(
        "--output-dir",
        type=str,
        help="User-specified L1 output file directory to override config file.",
    )
    args = argparser.parse_args()

    if args.input_L0_dir is not None:
        if not os.path.isdir(args.input_L0_dir):
            argparser.error(
                "--input-L0-dir error: "
                + str(args.input_L0_dir)
                + " not a valid directory"
            )

    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            argparser.error(
                "--output-dir error: " + str(args.output_dir) + " not a valid directory"
            )

    this_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    if args.conf_file is not None:
        if not os.path.isfile(args.conf_file):
            argparser.error(
                "--output-dir error: " + str(args.output_dir) + " not a valid file"
            )
        conf_file = Path(args.conf_file)
    else:
        conf_file = this_dir.joinpath(Path("../config.sh"))

    settings = {
        "L1_L0_INPUT": "",
        "L1_L1_OUTPUT": "",
        "DELETE_LO_FILE": "",
        "L1_LANDMASK": "",
        "L1_DEM": "",
        "L1_DTU": "",
        "L1_SV_PRN": "",
        "L1_SV_eirp": "",
        "L1_DICT": "",
        "L1a_CAL_COUNTS": "",
        "L1a_CAL_POWER": "",
        "L1_LANDCOVER": "",
        "L1_LHCP_L": "",
        "L1_LHCP_R": "",
        "L1_RHCP_L": "",
        "L1_RHCP_R": "",
        "L1_CDDIS_USERNAME": "",
        "L1_CDDIS_PASSWORD": "",
        "AIRCRAFT_REG": "",
        "DDM_SOURCE": "",
        "DDM_TIME_TYPE_SELECTOR": "",
        "DEM_SOURCE": "",
        "L1_ALGORITHM_VERSION": "",
        "L1_DATA_VERSION": "",
        "L1A_SIG_LUT_VERSION": "",
        "L1A_NOISE_LUT_VERSION": "",
        "A_LUT_VERSION": "",
        "NGRX_PORT_MAPPING_VERSION": "",
        "NADIR_ANT_DATA_VERSION": "",
        "ZENITH_ANT_DATA_VERSION": "",
        "PRN_SV_MAPS_VERSION": "",
        "GPS_EIRP_PARAM_VERSION": "",
        "LAND_MASK_VERSION": "",
        "SURFACE_TYPE_VERSION": "",
        "MEAN_SEA_SURFACE_VERSION": "",
        "PER_BIN_ANT_VERSION": "",
        "CONVENTIONS": "",
        "TITLE": "",
        "HISTORY": "",
        "STANDARD_NAME_VOCABULARY": "",
        "COMMENT": "",
        "PROCESSING_LEVEL": "",
        "CREATOR_TYPE": "",
        "INSTITUTION": "",
        "CREATOR_NAME": "",
        "PUBLISHER_NAME": "",
        "PUBLISHER_EMAIL": "",
        "PUBLISHER_URL": "",
        "GEOSPATIAL_LAT_MIN": "",
        "GEOSPATIAL_LAT_MAX": "",
        "GEOSPATIAL_LON_MIN": "",
        "GEOSPATIAL_LON_MAX": "",
    }

    with open(conf_file) as f:
        for line in f:
            if line.split("=")[0] in settings.keys():
                settings[line.split("=")[0]] = (
                    line.split("=")[1].replace("\n", "").replace('"', "")
                )
    if not all(
        [y for x, y in settings.items() if x not in ["L1_L0_INPUT", "L1_L1_OUTPUT"]]
    ):
        missing = [x for x, y in settings.items() if not y]
        raise Exception(
            "config file missing the following variables: " + ", ".join(missing)
        )

    if (settings["L1_CDDIS_USERNAME"] == "USERNAME") or (
        settings["L1_CDDIS_PASSWORD"] == "PASSWORD"
    ):
        raise Exception(
            "Please set L1_CDDIS credentials correctly or risk retrieving no orbit files"
        )

    if args.input_L0_dir is not None:
        L0_path = Path(args.input_L0_dir)
    elif settings["L1_L0_INPUT"]:
        # probably add some logic here to handle relative vs explicit paths in settings
        L0_path = Path(settings["L1_L0_INPUT"])
    else:
        L0_path = this_dir.joinpath(Path("../dat/raw/"))

    if args.output_dir is not None:
        L1_path = Path(args.output_dir)
    elif settings["L1_L1_OUTPUT"]:
        # probably add some logic here to handle relative vs explicit paths in settings
        L1_path = Path(settings["L1_L1_OUTPUT"])
    else:
        L1_path = this_dir.joinpath(Path("../out/"))

    # Hardcoded directories as locations for input files
    A_phy_LUT_path = this_dir.joinpath(Path("../dat/A_phy_LUT/"))
    landmask_path = this_dir.joinpath(Path("../dat/cst/"))
    dem_path = this_dir.joinpath(Path("../dat/dem/"))
    dtu_path = this_dir.joinpath(Path("../dat/dtu/"))
    gps_path = this_dir.joinpath(Path("../dat/gps/"))
    L1_dict_path = this_dir.joinpath(Path("../dat/L1_Dict/"))
    L1a_path = this_dir.joinpath(Path("../dat/L1a_cal/"))
    lcv_path = this_dir.joinpath(Path("../dat/lcv/"))
    orbit_path = this_dir.joinpath(Path("../dat/orbits/"))
    pek_path = this_dir.joinpath(Path("../dat/pek/"))
    rng_path = this_dir.joinpath(Path("../dat/rng/"))

    # L1_A_PHY_LUT = A_phy_LUT_path.joinpath(Path(settings["L1_A_PHY_LUT"]))
    # load ocean/land (distance to coast) mask
    L1_LANDMASK = landmask_path.joinpath(Path(settings["L1_LANDMASK"]))
    # load SRTM_30 DEM
    L1_DEM = dem_path.joinpath(Path(settings["L1_DEM"]))
    # load DTU10 model
    L1_DTU = dtu_path.joinpath(Path(settings["L1_DTU"]))
    # load PRN-SV and SV-EIRP(static) LUT
    L1_SV_PRN = gps_path.joinpath(Path(settings["L1_SV_PRN"]))
    L1_SV_eirp = gps_path.joinpath(Path(settings["L1_SV_eirp"]))
    # L1_DICT file
    L1_DICT = L1_dict_path.joinpath(Path(settings["L1_DICT"]))
    # load L1a calibration tables
    L1a_CAL_COUNTS = L1a_path.joinpath(Path(settings["L1a_CAL_COUNTS"]))
    L1a_CAL_POWER = L1a_path.joinpath(Path(settings["L1a_CAL_POWER"]))
    # load landcover mask
    L1_LANDCOVER = lcv_path.joinpath(Path(settings["L1_LANDCOVER"]))
    # process inland water mask
    water_mask_paths = ["160E_40S", "170E_30S", "170E_40S"]
    # load and process nadir NGRx-GNSS antenna patterns
    L1_LHCP_L = rng_path.joinpath(Path(settings["L1_LHCP_L"]))
    L1_LHCP_R = rng_path.joinpath(Path(settings["L1_LHCP_R"]))
    L1_RHCP_L = rng_path.joinpath(Path(settings["L1_RHCP_L"]))
    L1_RHCP_R = rng_path.joinpath(Path(settings["L1_RHCP_R"]))
    rng_filenames = [L1_LHCP_L, L1_LHCP_R, L1_RHCP_L, L1_RHCP_R]

    # set up input class that holds all input file data
    inp = input_files(
        L1a_CAL_COUNTS,
        L1a_CAL_POWER,
        L1_DEM,
        L1_DTU,
        L1_LANDMASK,
        L1_LANDCOVER,
        water_mask_paths,
        pek_path,
        L1_SV_PRN,
        L1_SV_eirp,
        rng_filenames,
        A_phy_LUT_path,
        orbit_path,
    )

    # find all L0 files in L0_path
    L0_files = [filepath for filepath in L0_path.glob("*.nc")]

    for filepath in sorted(L0_files):
        # print(filepath, os.path.basename(filepath))
        new_L1_file = os.path.basename(filepath).split(".")
        new_L1_file = new_L1_file[0] + "_L1." + new_L1_file[1]
        new_L1_file = L1_path.joinpath(Path(new_L1_file))
        try:
            process_L1s(filepath, new_L1_file, inp, L1_DICT, settings)
            # This flag is loaded as a string, so cast back to int to ensure it
            # is evaluated correctly!
            if int(settings["DELETE_LO_FILE"]):
                os.remove(str(filepath))
        except OrbitFileDelayError as exc:
            # print OrbitFileDelayError, but otherwise just skip this
            # file until next time
            print(exc)
            continue
