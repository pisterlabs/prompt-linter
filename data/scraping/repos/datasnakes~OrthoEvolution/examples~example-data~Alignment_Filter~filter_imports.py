import os
from pathlib import Path
from shutil import copy
from guidance2 import Guidance2Commandline
from pal2nal import Pal2NalCommandline
from Bio import SeqIO
from utils import multi_fasta_manipulator
import subprocess