#!/usr/bin/env python
__author__ = 'Sergei F. Kliver'
import argparse
from RouToolPa.Tools.MultipleAlignment import GUIDANCE2
from RouToolPa.GeneralRoutines import FileRoutines
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", action="store", dest="input", required=True,
                    type=lambda x: FileRoutines.make_list_of_path_to_files(x.split(",")),
                    help="Comma-separated list of files or directory with files containing sequences to be aligned")
parser.add_argument("-p", "--processes", action="store", dest="processes", type=int, default=1,
                    help="Number of simultaneously running alignments")
parser.add_argument("-o", "--output_directory", action="store", dest="output",
                    type=FileRoutines.check_path,
                    required=True,
                    help="Output directory")
parser.add_argument("-d", "--handling_mode", action="store", dest="handling_mode", default="local",
                    help="Handling mode. Allowed: local(default), slurm")
parser.add_argument("-j", "--slurm_job_name", action="store", dest="slurm_job_name", default="JOB",
                    help="Slurm job name. Default: JOB")
parser.add_argument("-m", "--slurm_max_jobs", action="store", dest="slurm_max_jobs", default=1000, type=int,
                    help="Slurm max jobs. Default: 1000")
parser.add_argument("-y", "--slurm_log_prefix", action="store", dest="slurm_log_prefix",
                    help="Slurm log prefix. ")
parser.add_argument("-z", "--slurm_cmd_log_file", action="store", dest="slurm_cmd_log_file",
                    help="Slurm cmd logfile")
parser.add_argument("-l", "--slurm_error_log_prefix", action="store", dest="slurm_error_log_prefix",
                    help="Slurm error log prefix")
parser.add_argument("-e", "--max_memory_per_task", action="store", dest="max_memory_per_task", default="5000",
                    help="Maximum memory per task in megabytes. Default: 5000")
parser.add_argument("-a", "--slurm_max_running_time", action="store", dest="slurm_max_running_time",
                    default="100:00:00",
                    help="Slurm max running time in hh:mm:ss format. Default: 100:00:00")
parser.add_argument("-w", "--slurm_modules_list", action="store", dest="slurm_modules_list", default=[],
                    type=lambda s: s.split(","),
                    help="Comma-separated list of modules to load. Set modules for hmmer and python")
parser.add_argument("-g", "--guidance_dir", action="store", dest="guidance_dir", default="",
                    help="Path to directory with GUIDANCE2 script")
parser.add_argument("--mafft_bin", action="store", dest="mafft_bin",
                    help="Path to MAFFT binary. ")
parser.add_argument("--prank_bin", action="store", dest="prank_bin",
                    help="Path to PRANK binary. ")

args = parser.parse_args()

GUIDANCE2.threads = args.processes
GUIDANCE2.path = args.guidance_dir
GUIDANCE2.parallel_codon_alignment(args.input, args.output, msa_tool='prank',
                                   bootstrap_number=100, genetic_code=1,
                                   threads=None,
                                   msa_tool_options=None, seq_cutoff=None,
                                   col_cutoff=None, mafft_bin=args.mafft_bin,
                                   prank_bin=args.prank_bin,
                                   muscle_bin=None, pagan_bin=None,
                                   ruby_bin=None, program=None,
                                   cmd_log_file=args.slurm_cmd_log_file,
                                   cpus_per_task=1,
                                   handling_mode=args.handling_mode,
                                   job_name=args.slurm_job_name,
                                   log_prefix=args.slurm_log_prefix,
                                   error_log_prefix=args.slurm_error_log_prefix,
                                   max_jobs=args.slurm_max_jobs,
                                   max_running_time=args.slurm_max_running_time,
                                   max_memory_per_node=args.max_memory_per_task,
                                   max_memmory_per_cpu=None,
                                   modules_list=args.slurm_modules_list,
                                   environment_variables_dict=None)
