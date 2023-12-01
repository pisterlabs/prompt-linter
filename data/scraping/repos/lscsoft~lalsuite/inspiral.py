"""
Classes needed for the inspiral analysis pipeline.
This script produced the necessary condor submit and dag files to run
the standalone inspiral code on LIGO data
"""

from __future__ import print_function

__author__ = 'Duncan Brown <duncan@gravity.phys.uwm.edu>'
__date__ = '$Date$'
__version__ = '$Revision$'

import copy
import string
import sys, os, re, subprocess
from glue import pipeline
from glue import lal


class InspiralError(Exception):
  def __init__(self, args=None):
    self.args = args


#############################################################################


class InspiralAnalysisJob(pipeline.AnalysisJob, pipeline.CondorDAGJob):
  """
  An inspiral analysis job captures some of the common features of the specific
  inspiral jobs that appear below.  Specifically, the universe and exec_name
  are set, the stdout and stderr from the job are directed to the logs
  directory. The path to the executable is determined from the ini file.
  """
  def __init__(self,cp,sections,exec_name,extension='xml'):
    """
    cp = ConfigParser object from which options are read.
    sections = sections of the ConfigParser that get added to the opts
    exec_name = exec_name name in ConfigParser
    """
    self.__exec_name = exec_name
    self.__extension = extension
    universe = cp.get('condor','universe')
    executable = cp.get('condor',exec_name)
    pipeline.CondorDAGJob.__init__(self,universe,executable)
    pipeline.AnalysisJob.__init__(self,cp)
    self.add_condor_cmd('copy_to_spool','False')
    self.__use_gpus = cp.has_option('condor', 'use-gpus')

    mycp = copy.deepcopy(cp)
    for sec in sections:
      if mycp.has_section(sec):
        # add all the other options as arguments to the code
        self.add_ini_opts(mycp, sec)

      else:
        print("warning: config file is missing section [" + sec + "]", file=sys.stderr)

    self.set_stdout_file('logs/' + exec_name + \
        '-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/' + exec_name + \
        '-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file(exec_name + '.sub')

  def set_exec_name(self,exec_name):
    """
    Set the exec_name name
    """
    self.__exec_name = exec_name

  def get_exec_name(self):
    """
    Get the exec_name name
    """
    return self.__exec_name

  def set_extension(self,extension):
    """
    Set the file extension
    """
    self.__extension = extension

  def get_extension(self):
    """
    Get the extension for the file name
    """
    return self.__extension

  def get_use_gpus(self):
    """
    Get whether this job was requested to run on a GPU node
    """
    return self.__use_gpus


#############################################################################

class InspiralPlottingJob(InspiralAnalysisJob):
  """
  The InspiralPlottingJob class will assign options common to all plotting
  jobs. Currently this is only MPLCONFIGDIR.
  """
  def __init__(self,cp,sections,exec_name,extension='xml'):
    """
    cp = ConfigParser object from which options are read.
    sections = sections of the ConfigParser that get added to the opts
    exec_name = exec_name name in ConfigParser
    """
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('getenv','True')
    if cp.has_option('pipeline','matplotlibdir'):
      MPLConfigPath = cp.get('pipeline','matplotlibdir')
      self.add_condor_cmd('environment','MPLCONFIGDIR=' + MPLConfigPath)

#############################################################################

class TmpltBankJob(InspiralAnalysisJob):
  """
  A lalapps_tmpltbank job used by the inspiral pipeline. The static options
  are read from the sections [data] and [tmpltbank] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The job
  runs in the universe specfied in the ini file. The path to the executable
  is determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'tmpltbank'
    extension = 'xml'
    sections = ['data','tmpltbank']

    have_pycbc = False
    if cp.has_option('tmpltbank', 'pycbc'):
         have_pycbc=True
         cp.set('condor', 'universe', 'vanilla')

    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)

    if have_pycbc:
        self.add_condor_cmd('getenv', 'True')



class InspInjJob(InspiralAnalysisJob):
  """
  A lalapps_inspinj job used by the grb inspiral pipeline. The static options
  are read from the section [inspinj] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The
  job runs in the universe specified in the ini file. The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'inspinj'
    sections = ['inspinj']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    self.set_universe('vanilla')

    self.__listDone=[]
    self.__listNodes=[]

  def set_done(self,number,node):
    self.__listDone.append(number)
    self.__listNodes.append(node)

  def check_node(self, number):
    if self.__listDone.count(number):
      index=self.__listDone.index(number)
      return self.__listNodes[index]
    return None


class BbhInjJob(InspiralAnalysisJob):
  """
  A lalapps_bbhinj job used by the online inspiral pipeline. The static options
  are read from the section [bbhinj] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The
  job runs in the universe specified in the ini file. The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'bbhinj'
    sections = ['bbhinj']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class RandomBankJob(InspiralAnalysisJob):
  """
  A lalapps_randombank job used by the inspiral pipeline. The static options
  are read from the section [randombank] in the ini file. The stdout and
  stderr from the job are directed to the logs directory. The job runs in the
  universe specfied in the ini file. The path to the executable is determined
  from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'randombank'
    sections = ['randombank']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class SplitBankJob(InspiralAnalysisJob):
  """
  A lalapps_splitbank job used by the inspiral pipeline. The static options
  are read from the section [splitbank] in the ini file. The stdout and stderr
  from the job are directed to the logs directory. The job runs in the
  universe specfied in the ini file. The path to the executable is determined
  from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'splitbank'
    sections = ['splitbank']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class InspiralJob(InspiralAnalysisJob):
  """
  A lalapps_inspiral job used by the inspiral pipeline. The static options
  are read from the sections [data] and [inspiral] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The job
  runs in the universe specfied in the ini file. The path to the executable
  is determined from the ini file. If the user requested GPU utilization,
  checks are done to ensure a successful run and the necessary Condor
  commands are added.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'inspiral'
    sections = ['data','inspiral']
    extension = 'xml'

    have_pycbc = False
    if cp.has_option('inspiral', 'pycbc'):
         have_pycbc=True
         cp.set('condor', 'universe', 'vanilla')
         cp.remove_option('inspiral', 'pycbc')

    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")
    self.add_condor_cmd('request_memory', '1000')

    if have_pycbc:
        self.add_condor_cmd('getenv', 'True')


    if self.get_use_gpus():
      # make sure the vanilla universe is being used
      universe = cp.get('condor', 'universe')
      if universe != 'vanilla':
        raise RuntimeError('Cannot run GPU inspiral jobs on Condor ' + \
            universe + ' universe. Please use vanilla.')
      # make sure the executable has CUDA dependencies
      executable = cp.get('condor', exec_name)
      objdump_re = re.compile(r'^\s*NEEDED\s*(libcufft\.|libcudart\.).*')
      proc = subprocess.Popen(['objdump', '-p', executable], \
          stdin=None, stdout=subprocess.PIPE)
      cuda_deps = False
      for line in proc.stdout:
        m = objdump_re.match(line)
        if m:
          cuda_deps = True
          break
      if not cuda_deps:
        raise RuntimeError('Inspiral executable has no CUDA ' + \
            'dependencies. Please use a CUDA-enabled build.')
      self.add_opt('gpu-device-id', '0')
      self.add_condor_cmd('+WantGPU', 'true')
      self.add_condor_cmd('Requirements', '( GPU_PRESENT =?= true)')


class InspiralCkptJob(InspiralAnalysisJob):
  """
  A lalapps_inspiral job used by the inspiral pipeline. The static options
  are read from the sections [data] and [inspiral] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The job
  runs in the universe specfied in the ini file. The path to the executable
  is determined from the ini file.
  This one checkpoints.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'inspiral'
    sections = []
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    self.add_short_opt('_condor_relocatable', '')


class PTFInspiralJob(InspiralAnalysisJob):
  """
  A lalapps_inspiral job used by the inspiral pipeline. The static options
  are read from the sections [data] and [inspiral] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The job
  runs in the universe specfied in the ini file. The path to the executable
  is determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'coh_PTF_inspiral'
    sections = ['coh_PTF_inspiral']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    ramValue = 1390
    if cp.has_section('coh_PTF_inspiral-meta'):
      if cp.has_option('coh_PTF_inspiral-meta','minimum-ram'):
        ramValue = int(cp.get('coh_PTF_inspiral-meta','minimum-ram'))
    self.add_condor_cmd('request_memory', '%d' %(ramValue))


class TrigbankJob(InspiralAnalysisJob):
  """
  A lalapps_trigbank job used by the inspiral pipeline. The static
  options are read from the section [trigbank] in the ini file.  The
  stdout and stderr from the job are directed to the logs directory. The job
  always runs in the scheduler universe. The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'trigbank'
    sections = ['trigbank']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class IncaJob(InspiralAnalysisJob):
  """
  A lalapps_inca job used by the inspiral pipeline. The static options are
  read from the section [inca] in the ini file.  The stdout and stderr from
  the job are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'inca'
    sections = ['inca']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class ThincaJob(InspiralAnalysisJob):
  """
  A lalapps_thinca job used by the inspiral pipeline. The static options are
  read from the section [thinca] in the ini file.  The stdout and stderr from
  the job are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    exec_name = 'thinca'
    #sections = ['thinca']
    sections = []
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    if cp.has_section('thinca'):
      self.add_ini_opts(cp,'thinca')


  def add_ini_opts(self, cp, section):
    """
    Parse command line options from a given section in an ini file and
    pass to the executable.
    @param cp: ConfigParser object pointing to the ini file.
    @param section: section of the ini file to add to the options.
    """
    for opt in cp.options(section):
      arg = string.strip(cp.get(section,opt))
      #self.add_opt(opt,arg)
      if opt[-4:] == "file":
        fname = os.path.split(arg)[-1]
        if fname not in os.listdir('.'):
          try:
            os.symlink(arg,os.path.split(arg)[-1])
            self.add_file_opt(opt,fname)
          except:
            print("sym link failed for " + arg + " grid workflows might be broken", file=sys.stderr)
            self.add_file_opt(opt,arg)
        else:
          self.add_file_opt(opt,fname)
      else:
        self.add_opt(opt,arg)


class ThincaToCoincJob(InspiralAnalysisJob):
  """
  A ThincaToCoinc job. The static options are read from the
  section [thinca_to_coinc] in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'thinca_to_coinc'
    sections = ['thinca_to_coinc']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self, cp, sections, exec_name, extension)
    self.add_condor_cmd('getenv', 'True')
    self.__experiment_start_time = None
    self.__experiment_end_time = None
    # overwrite standard log file names
    self.set_stdout_file('logs/' + exec_name + '-$(cluster)-$(process).out')
    self.set_stderr_file('logs/' + exec_name + '-$(cluster)-$(process).err')

  def set_experiment_start_time(self, experiment_start_time):
    """
    Sets the experiment-start-time option. This is a required option.
    @param experiment_start_time: gps start time of the experiment the thinca_to_coinc
    job is in.
    """
    self.add_opt('experiment-start-time', experiment_start_time)
    self.__experiment_start_time = experiment_start_time

  def set_experiment_end_time(self, experiment_end_time):
    """
    Sets the experiment-end-time option. This is a required option.
    @param experiment_end_time: gps end time of the experiment the thinca_to_coinc
    job is in.
    """
    self.add_opt('experiment-end-time', experiment_end_time)
    self.__experiment_end_time = experiment_end_time

  def get_experiment_start_time(self, experiment_start_time):
    """
    Returns the value of the experiment-start-time option.
    """
    return self.__experiment_start_time

  def get_experiment_end_time(self, experiment_end_time):
    """
    Returns the value of the experiment-end-time option.
    """
    return self.__experiment_start_time

  def set_simulation(self):
    """
    Adds the simulation argument to the job.
    """
    self.add_opt('simulation', '')

class HWinjPageJob(InspiralAnalysisJob):
  """
  A HWinjPageJob, runs the hardware injection page script on the
  output of the pipeline
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = "hardware_inj_page"
    universe = "vanilla"
    sections = "[hardware-injection-page]"
    extension = 'html'
    executable = cp.get('condor',exec_name)
    pipeline.CondorDAGJob.__init__(self, universe, executable)
    pipeline.AnalysisJob.__init__(self, cp)
    self.add_condor_cmd('getenv','True')
    self.set_stdout_file('logs/' + exec_name + '-$(cluster)-$(process).out')
    self.set_stderr_file('logs/' + exec_name + '-$(cluster)-$(process).err')
    self.set_sub_file(exec_name + '.sub')

class SireJob(InspiralAnalysisJob):
  """
  A lalapps_sire job used by the inspiral pipeline. The stdout and stderr from
  the job are directed to the logs directory. The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'sire'
    sections = ['sire']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)

    # sire currently doesn't take GPS start/end times
    self.set_stdout_file('logs/sire-$(macroifo)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/sire-$(macroifo)-$(cluster)-$(process).err')


class CoireJob(InspiralAnalysisJob):
  """
  A lalapps_coire job used by the inspiral pipeline. The stdout and stderr from
  the job are directed to the logs directory. The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'coire'
    sections = ['coire']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)

    # coire currently doesn't take GPS start/end times
    self.set_stdout_file('logs/coire-$(macroifo)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/coire-$(macroifo)-$(cluster)-$(process).err')


class FrJoinJob(InspiralAnalysisJob):
  """
  A lalapps_frjoin job used by the inspiral pipeline. The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'frjoin'
    sections = []
    extension = 'gwf'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)

    # frjoin currently doesn't take GPS start/end times
    self.set_stdout_file('logs/frjoin-$(cluster)-$(process).out')
    self.set_stderr_file('logs/frjoin-$(cluster)-$(process).err')


class CohBankJob(InspiralAnalysisJob):
  """
  A lalapps_coherent_inspiral job used by the inspiral pipeline. The static
  options are read from the section [cohbank] in the ini file.  The stdout and
  stderr from the job are directed to the logs directory.  The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'cohbank'
    sections = ['cohbank']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class InspiralCoherentJob(InspiralAnalysisJob):
  """
  A lalapps_inspiral job used by the inspiral pipeline. The static options
  are read from the sections [data] and [inspiral] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The job
  runs in the universe specfied in the ini file. The path to the executable
  is determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'inspiral'
    sections = ['data']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")


class CohInspBankJob(InspiralAnalysisJob):
  """
  A lalapps_coherent_inspiral job used by the inspiral pipeline. The static
  options are read from the section [cohinspbank] in the ini file.  The stdout and
  stderr from the job are directed to the logs directory.  The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'cohinspbank'
    sections = ['cohinspbank']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class ChiaJob(InspiralAnalysisJob):
  """
  A lalapps_coherent_inspiral job used by the inspiral pipeline. The static
  options are read from the section [chia] in the ini file.  The stdout and
  stderr from the job are directed to the logs directory.  The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'chia'
    sections = ['chia']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)


class CohireJob(InspiralAnalysisJob):
  """
  A lalapps_cohire job used by the inspiral pipeline. The stdout and stderr from
  the job are directed to the logs directory. The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'cohire'
    sections = ['cohire']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)

    # cohire currently doesn't take GPS start/end times
    self.set_stdout_file('logs/cohire-$(macroifo)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/cohire-$(macroifo)-$(cluster)-$(process).err')


class InjFindJob(InspiralAnalysisJob):
  """
  An injfind job. The static options are read from the [injfind]
  section in the cp file.
  """
  def __init__(self, cp):
    """
    @param cp: a ConfigParser object from which the options are read.
    """
    exec_name = 'injfind'
    sections = ['injfind']
    extension = 'xml'
    InspiralAnalysisJob.__init__(self, cp, sections, exec_name, extension)
    self.add_condor_cmd('getenv', 'True')
    # overwrite standard log file names
    self.set_stdout_file('logs/' + exec_name + '-$(cluster)-$(process).out')
    self.set_stderr_file('logs/' + exec_name + '-$(cluster)-$(process).err')


#############################################################################


class InspiralAnalysisNode(pipeline.AnalysisNode, pipeline.CondorDAGNode):
  """
  An InspiralNode runs an instance of the inspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspiral.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    opts = job.get_opts()

    if ("pad-data" in opts) and int(opts['pad-data']):
      self.set_pad_data(int(opts['pad-data']))

    self.__zip_output = ("write-compress" in opts)
    self.__data_checkpoint = False

  def set_zip_output(self,zip):
    """
    Set the zip output flag
    """
    self.__zip_output = zip

  def get_zip_output(self):
    """
    Set the zip output flag
    """
    return self.__zip_output

  def get_output_base(self):
    """
    Returns the base file name of output from the inspiral code. This is
    assumed to follow the standard naming convention:

    IFO-EXECUTABLE_IFOTAG_USERTAG-GPS_START-DURATION
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo():
      raise InspiralError("Start time, end time or ifo has not been set")

    filebase = self.get_ifo() + '-' + self.job().get_exec_name().upper()

    if self.get_ifo_tag():
      filebase += '_' + self.get_ifo_tag()
    if self.get_user_tag():
      filebase += '_' + self.get_user_tag()

    filebase +=  '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start())

    return(filebase)

  def get_output(self):
    """
    Returns the file name of output from the inspiral code. This is obtained
    from the get_output_base() method, with the correct extension added.
    """
    filename = self.get_output_base()
    filename += '.' + self.job().get_extension()

    if self.get_zip_output():
      filename += '.gz'

    if self.__data_checkpoint is False:
      self.add_output_file(filename)

    return filename

  def get_checkpoint_image(self):
    """
    Returns the file name of condor checkpoint from the inspiral code. This is
    obtained from the get_output_base() method, with the correct extension added.
    """
    filename = self.get_output_base()
    filename += '.ckpt'

    self.add_output_file(filename)

    return filename

  def set_data_checkpoint(self):
    """
    Lets it be known that data checkpointing exists
    """
    self.add_var_opt('data-checkpoint','')
    self.__data_checkpoint = True

  def get_data_checkpoint(self):
    """
    Lets it be known that data checkpointing exists
    """
    return self.__data_checkpoint

  def get_output_cache(self):
    """
    Returns the name of the cache file output from the inspiral analysis codes.
    This is obtained from the get_output_base() method, with the correct
    extension added.
    """
    filename = self.get_output_base()
    filename += '.cache'

  def get_froutput(self):
    """
    Returns the file name of output frame from the inspiral code.
    """
    gwffile = self.get_output_base()
    gwffile += '.gwf'

    self.add_output_file(gwffile)

    return gwffile

  def finalize(self):
    """
    set the data_start_time and data_end_time
    """
    if self.get_pad_data():
      self.set_data_start(self.get_start() - \
          self.get_pad_data())
      self.set_data_end(self.get_end() + \
          self.get_pad_data())

#############################################################################

class InspiralPlottingNode(InspiralAnalysisNode):
  """
  An InspiralPlottingNode runas an instance of the inspiral plotting code in
  a Condor Dag
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of the plotting code
    """
    InspiralAnalysisNode.__init__(self,job)

#############################################################################

class InspInjNode(InspiralAnalysisNode):
  """
  A InspInjNode runs an instance of the inspinj generation job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspinj.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__outputName = None
    self.__seed = None

  def set_seed(self,seed):
    """
    Set the seed of the injection file by setting a --seed option to the
    node when it is executed.
    @param seed: seed of the job
    """
    self.add_var_opt('seed',seed)
    self.__seed = seed

  def get_seed(self):
    """
    return the seed
    """
    return( self.__seed)

  def set_output(self, outputName):
    """
    Set the output name of the injection file
    @param outputName: name of the injection file created
    """
    self.add_var_opt('output',outputName)
    self.__outputName = outputName

  def get_output(self):
    """
    Return the manually-set output name if it exists, otherwise, derive the
    name like other InspiralAnalysisNodes.
    """
    if self.__outputName:
      self.add_output_file(self.__outputName)
      return self.__outputName
    else:
      outputFile = "HL-INJECTIONS_" + str(self.get_seed())
      if self.get_user_tag():
        outputFile += "_" + self.get_user_tag()
      outputFile += "-" + str(self.get_start()) + "-" + str(self.get_end() - \
          self.get_start()) + ".xml"
      self.add_output_file(outputFile)
      return(outputFile)


class BbhInjNode(InspiralAnalysisNode):
  """
  A BbhInjNode runs an instance of the bbhinj generation job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_bbhinj.
    """
    InspiralAnalysisNode.__init__(self,job)

  def set_seed(self,seed):
    """
    Set the seed of the injection file by setting a --seed option to the
    node when it is executed.
    @param seed: seed of the job
    """
    self.add_var_opt('seed',seed)
    self.__seed = seed

  def get_output(self):
    """
    Returns the file name of output from the injection generation code. This
    must be kept synchronized with the name of the output file in bbhinj.c.
    """
    if not self.get_start() or not self.get_end():
      raise InspiralError("Start time or end time has not been set")
    if self.get_user_tag():
      bbhinject = 'HL-INJECTIONS_' + self.get_user_tag() + '-'
      bbhinject = bbhinject + str(self.get_start()) + '-'
      bbhinject = bbhinject + str(self.get_end()-self.get_start()) + '.xml'
    elif self.__seed:
      bbhinject = 'HL-INJECTIONS_' + str(self.__seed) + '-'
      bbhinject = bbhinject + str(self.get_start()) + '-'
      bbhinject = bbhinject + str(self.get_end()-self.get_start()) + '.xml'
    else:
      bbhinject = 'HL-INJECTIONS-' + str(self.get_start()) + '-'
      bbhinject = bbhinject + str(self.get_end()-self.get_start()) + '.xml'

    self.add_output_file(bbhinject)

    return bbhinject


class TmpltBankNode(InspiralAnalysisNode):
  """
  A TmpltBankNode runs an instance of the template bank generation job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_tmpltbank.
    """
    InspiralAnalysisNode.__init__(self,job)


class RandomBankNode(InspiralAnalysisNode):
  """
  A RandomBankNode runs an instance of the random bank generation job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_randombank.
    """
    InspiralAnalysisNode.__init__(self,job)

  def get_output(self):
    """
    Returns the file name of output from the template bank code. This must
    be kept synchronized with the name of the output file in randombank.c.
    """
    if not self.get_start() or not self.get_end():
      raise InspiralError("Start time or end time has not been set")
    if self.get_user_tag():
      bank = 'P-TMPLTBANK_' + self.get_user_tag() + '-'
      bank = bank + str(self.get_start())
    else:
      bank = 'P-TMPLTBANK-' + str(self.get_start())
    bank = bank + '-' + str(self.get_end() - self.get_start()) + '.xml'

    self.add_output_file(bank)

    return bank


class SplitBankNode(InspiralAnalysisNode):
  """
  A SplitBankNode runs an instance of the split template bank job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_tmpltbank.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__bankfile = None
    self.__numbanks = None

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)
    self.__bankfile = bank

  def get_bank(self):
    return self.__bankfile

  def set_num_banks(self,numbanks):
    self.add_var_opt('number-of-banks',numbanks)
    self.__numbanks = int(numbanks)

  def get_num_banks(self):
    return self.__numbanks

  def get_output(self):
    """
    Returns a list of the file names of split banks. This must be kept
    synchronized with the name of the output files in splitbank.c.
    """
    if not self.get_bank() or not self.get_num_banks():
      raise InspiralError("Bank file or number of banks has not been set")

    banks = []
    x = self.__bankfile.split('-')
    for i in range( 0, int(self.get_num_banks()) ):
      banks.append("%s-%s_%2.2d-%s-%s" % (x[0], x[1], i, x[2], x[3]))

    return banks


class InspiralNode(InspiralAnalysisNode):
  """
  An InspiralNode runs an instance of the inspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspiral.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__injections = None

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)

  def set_injections(self, injections):
    """
    Set the injection file for this node
    """
    self.__injections = injections
    self.add_var_opt('injection-file', injections)
    self.add_input_file(injections)

  def get_injections(self):
    """
    Returns the injection file
    """
    return self.__injections


class InspiralCkptNode(InspiralAnalysisNode):
  """
  An InspiralCkptNode runs an instance of the inspiral code in a Condor DAG.
  It then checkpoints to a file. This sets the file.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspiral.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__outfile = None
    self.__injections = None

  def set_output(self, outfile):
    """
    Sets a placeholder output name which we can use to pass through the
    name of the output file from the original inspiral job.
    """
    self.__outfile = outfile

  def get_output(self):
    """
    Returns the filename from set_output().
    """
    if self.__outfile:
      self.add_output_file(self.__outfile)
    return self.__outfile

  def set_injections(self, injections):
    """
    Set the injection file for this node
    """
    self.__injections = injections

  def get_injections(self):
    """
    Returns the injection file
    """
    return self.__injections

  def set_checkpoint_image(self, ckptin):
    """
    Adds the argument -_condor_restart for the
    cases in which we want to checkpoint and grabs the
    checkpoint image name from get_checkpoint_image in
    the InspiralAnalysisCode section.
    """
    self.add_input_file(ckptin)
    self.add_var_opt('_condor_restart', ckptin, short=True)


class PTFInspiralNode(InspiralAnalysisNode):
  """
  An InspiralNode runs an instance of the inspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspiral.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__injections = None
    self.set_zip_output(True)

  def set_spin_bank(self,bank):
    self.add_var_opt('spin-bank', bank)
    self.add_input_file(bank)

  def set_no_spin_bank(self,bank):
    self.add_var_opt('non-spin-bank',bank)
    self.add_input_file(bank)

  def set_output(self):
    self.add_var_opt('output-file',self.get_output_base()+ '.xml.gz')

  def set_injections(self, injections):
    """
    Set the injection file for this node
    """
    self.__injections = injections
    self.add_var_opt('injection-file', injections)
    self.add_input_file(injections)

  def get_injections(self):
    """
    Returns the injection file
    """
    return self.__injections

  def set_seed(self,seed):
    self.add_var_opt('random-seed',seed)


class PTFSpinCheckerNode(InspiralAnalysisNode):
  """
  An InspiralNode runs an instance of the inspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspiral.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__injections = None

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)

  def set_spin_output(self,spinBank):
    self.add_var_opt('spin-bank',spinBank)

  def set_nospin_output(self,noSpinBank):
    self.add_var_opt('non-spin-bank',noSpinBank)


class TrigbankNode(InspiralAnalysisNode):
  """
  A TrigbankNode runs an instance of the triggered bank generator in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of trigbank.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__input_ifo = None

  def set_input_ifo(self,ifo):
    self.add_var_opt('input-ifo', ifo)
    self.__input_ifo = ifo

  def get_input_ifo(self):
    return self.__input_ifo

  def set_output_ifo(self,ifo):
    self.add_var_opt('output-ifo', ifo)
    self.set_ifo(ifo)


class IncaNode(InspiralAnalysisNode):
  """
  An IncaNode runs an instance of the inspiral coincidence code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inca.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__ifo_a = None
    self.__ifo_b = None

  def set_ifo_a(self, ifo):
    """
    Set the interferometer code to use as IFO A.
    ifo = IFO code (e.g. L1, H1 or H2).
    """
    self.add_var_opt('ifo-a', ifo)
    self.__ifo_a = ifo

  def get_ifo_a(self):
    """
    Returns the IFO code of the primary interferometer.
    """
    return self.__ifo_a

  def set_ifo_b(self, ifo):
    """
    Set the interferometer code to use as IFO B.
    ifo = IFO code (e.g. L1, H1 or H2).
    """
    self.add_var_opt('ifo-b', ifo)
    self.__ifo_b = ifo

  def get_ifo_b(self):
    """
    Returns the IFO code of the primary interferometer.
    """
    return self.__ifo_b

  def get_output_a(self):
    """
    Returns the file name of output from inca for ifo a. This must be kept
    synchronized with the name of the output file in inca.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo_a():
      raise InspiralError("Start time, end time or ifo a has not been set")

    basename = self.get_ifo_a() + '-INCA'

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()
    if self.get_user_tag():
      basename += '_' + self.get_user_tag()

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.get_zip_output():
      filename += '.gz'

    self.add_output_file(filename)
    return filename

  def get_output(self):
    return self.get_output_a()

  def get_output_b(self):
    """
    Returns the file name of output from inca for ifo b. This must be kept
    synchronized with the name of the output file in inca.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo_b():
      raise InspiralError("Start time, end time or ifo a has not been set")

    basename = self.get_ifo_b() + '-INCA'

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()
    if self.get_user_tag():
      basename += '_' + self.get_user_tag()

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.get_zip_output():
      filename += '.gz'

    self.add_output_file(filename)
    return filename


class ThincaNode(InspiralAnalysisNode):
  """
  A ThincaNode runs an instance of the inspiral coincidence code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inca.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__ifo_g1 = None
    self.__ifo_h1 = None
    self.__ifo_h2 = None
    self.__ifo_l1 = None
    self.__ifo_t1 = None
    self.__ifo_v1 = None
    self.__num_slides = None

  def set_ifo(self, ifo, pass_to_command_line=True):
    """
    Add the interferometer to the list of ifos
    ifo = IFO code (e.g. G1,L1,V1,T1, H1 or H2).
    pass_to_command_line = boolean for adding ifo-triggers as a variable option
    """
    #FIXME: Once thinca no longer needs --IFO-triggers flags,
    # use AnalysisNode's set_ifos method
    if ifo == 'G1':
      if pass_to_command_line:
        self.add_var_opt('g1-triggers','')
      self.__ifo_g1 = 'G1'
    elif ifo == 'H1':
      if pass_to_command_line:
        self.add_var_opt('h1-triggers','')
      self.__ifo_h1 = 'H1'
    elif ifo == 'H2':
      if pass_to_command_line:
        self.add_var_opt('h2-triggers','')
      self.__ifo_h2 = 'H2'
    elif ifo == 'L1':
      if pass_to_command_line:
        self.add_var_opt('l1-triggers','')
      self.__ifo_l1 = 'L1'
    elif ifo == 'T1':
      if pass_to_command_line:
        self.add_var_opt('t1-triggers','')
      self.__ifo_t1 = 'T1'
    elif ifo == 'V1':
      if pass_to_command_line:
        self.add_var_opt('v1-triggers','')
      self.__ifo_v1 = 'V1'

  def get_ifo_g1(self):
    """
    Returns the IFO code of g1.
    """
    return self.__ifo_g1

  def get_ifo_h1(self):
    """
    Returns the IFO code of h1.
    """
    return self.__ifo_h1

  def get_ifo_h2(self):
    """
    Returns the IFO code of h2.
    """
    return self.__ifo_h2

  def get_ifo_l1(self):
    """
    Returns the IFO code of l1.
    """
    return self.__ifo_l1

  def get_ifo_t1(self):
    """
    Returns the IFO code of t1.
    """
    return self.__ifo_t1

  def get_ifo_v1(self):
    """
    Returns the IFO code of v1.
    """
    return self.__ifo_v1

  def get_ifos(self):
    """
    Returns the ordered list of ifos.
    """
    ifos = ''
    if self.get_ifo_g1():
      ifos += self.get_ifo_g1()
    if self.get_ifo_h1():
      ifos += self.get_ifo_h1()
    if self.get_ifo_h2():
      ifos += self.get_ifo_h2()
    if self.get_ifo_l1():
      ifos += self.get_ifo_l1()
    if self.get_ifo_t1():
      ifos += self.get_ifo_t1()
    if self.get_ifo_v1():
      ifos += self.get_ifo_v1()

    return ifos

  def set_num_slides(self, num_slides):
    """
    Set number of time slides to undertake
    """
    self.add_var_opt('num-slides',num_slides)
    self.__num_slides = num_slides

  def get_num_slides(self):
    """
    Returns the num_slides from .ini (>0 => time slides desired)
    """
    return self.__num_slides

  def get_output(self):
    """
    Returns the file name of output from thinca.  This must be kept
    synchronized with the name of the output file in thinca.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifos():
      raise InspiralError("Start time, end time or ifos have not been set")

    if self.__num_slides:
      basename = self.get_ifos() + '-' + self.job().get_exec_name().upper() \
          + '_SLIDE'
    else:
      basename = self.get_ifos() + '-' + self.job().get_exec_name().upper()

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()

    if self.get_user_tag():
      basename += '_' + self.get_user_tag()

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.get_zip_output():
      filename += '.gz'

    self.add_output_file(filename)
    return filename

class ThincaToCoincNode(InspiralAnalysisNode):
  """
  A ThincaToCoincNode runs an instance of a ThincaToCoincJob
  in a DAG.
  """
  def __init__(self, job):
    """
    @param job: A ThincaToCoincJob.
    """
    InspiralAnalysisNode.__init__(self, job)
    self.__input_cache = None
    self.__instruments = None
    self.__zero_lag_file = None
    self.__time_slide_file = None
    self.__veto_segments = None
    self.__veto_segments_name = None

  def set_input_cache(self, input_cache_name):
    """
    @param input_cache_name: cache file for thinca_to_coinc to
      read.
    """
    self.add_file_opt( 'ihope-cache', input_cache_name )
    self.__input_cache = input_cache_name

  def get_input_cache(self):
    """
    Returns input cache file for this node.
    """
    return self.__input_cache

  def get_output_from_cache(self, coinc_file_tag ):
    """
    Returns a list of files that this node will generate using the input_cache.
    The output file names are the same as the input urls, but with the
    zero_lag 'THINCA' file replaced with 'THINCA_TO_COINC', and with the
    filepaths  pointing to the current directory in which the
    thinca_to_coinc node is being run.
    """
    if not self.__input_cache:
      raise ValueError("no input-cache specified")
    # open the input cache file
    fp = open(self.__input_cache, 'r')
    input_cache = lal.Cache().fromfile(fp).sieve( description = coinc_file_tag )
    output_files = [ \
      '/'.join([ os.getcwd(),
      re.sub('INCA', 'INCA_TO_COINC', os.path.basename(entry.url)) ]) for entry in input_cache \
      ]
    return output_files

  def set_instruments(self, instruments):
    """
    @param instruments: instruments that are on for the
     THINCA files thinca_to_coinc is operating on.
    """
    self.add_var_opt('instruments', instruments)
    self.__instruments = instruments

  def get_instruments(self):
    """
    Returns instruments for this node.
    """
    return self.__instruments

  def set_veto_segments(self, veto_segments):
    """
    @param veto_segments: name of xml file containing the vetoes to apply
    """
    self.add_var_opt('veto-segments', veto_segments)
    self.__veto_segments = veto_segments

  def get_veto_segments(self):
    """
    Returns the name of the veto-segments file for this node.
    """
    return self.__veto_segments

  def set_veto_segments_name(self, veto_segments_name):
    """
    @param veto_segments_name: name of vetoes in the vetoes xml file to
    apply.
    """
    self.add_var_opt('veto-segments-name', veto_segments_name)
    self.__veto_segments_name = veto_segments_name

  def get_veto_segments_name(self):
    """
    Returns the name of the vetoes applied for this node.
    """
    return self.__veto_segments_name

  def set_zero_lag_file(self, zero_lag_file):
    """
    Sets zero_lag_file for input.
    """
    self.add_file_opt( 'zero-lag-file', zero_lag_file )
    self.__zero_lag_file = zero_lag_file

  def get_zero_lag_file(self):
    """
    Returns zero_lag_file.
    """
    return self.__zero_lag_file

  def set_time_slide_file(self, time_slide_file):
    """
    Sets the time_slide_file for input.
    """
    self.add_file_opt( 'time-slide-file', time_slide_file )
    self.__time_slide_file = time_slide_file

  def get_time_slide_file(self):
    """
    Returns the time_slide_file.
    """
    return self.__time_slide_file

class HWinjPageNode(InspiralAnalysisNode):
  """
  A HWinjPageNode runs an instance of a HWinjPageJob
  in a DAG.
  """
  def __init__(self, job):
    """
    @param job: A HWinjPageJob.
    """
    InspiralAnalysisNode.__init__(self, job)
    self.__input_cache = None
    self.__cache_string = None
    self.__outfile = None
    self.__segment_dir = None
    self.__source_xml = None

  def set_input_cache(self, input_cache_name):
    """
    @param input_cache_name: cache file for ligolw_cbc_hardware_inj_page
    to read.
    """
    self.add_var_opt('cache-file',input_cache_name)
    self.__input_cache = input_cache_name

  def set_source_xml(self, source_xml):
    """
    input_cache_name: cache file for ligolw_cbc_hardware_inj_page to read.
    """
    self.add_var_opt('source-xml',source_xml)
    self.__source_xml = source_xml

  def set_cache_string(self,cache_string):
    """
    @param cache_string: pattern to match files within cache
    """
    self.add_var_opt('cache-pattern',cache_string)
    self.__cache_string=cache_string

  def set_output_file(self,outfile_name):
    """
    @param outfile_name: Name of hw injection page
    """
    self.add_var_opt('outfile',outfile_name)
    self.__outfile=outfile_name

  def set_segment_dir(self,dir):
    """
    @param dir: directory in which to find hwinj segments
    """
    self.add_var_opt('segment-dir',dir)

class SireNode(InspiralAnalysisNode):
  """
  A SireNode runs an instance of the single inspiral reader code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_sire.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__injection_file = None
    self.__ifo_tag = None

  def set_ifo(self, ifo):
    """
    Add the list of interferometers
    """
    self.__ifo = ifo
    self.add_var_opt('ifo-cut',ifo)

  def get_ifo(self):
    """
    Returns the two letter IFO code for this node.
    """
    return self.__ifo

  def set_inj_file(self, file):
    """
    Sets the injection file
    """
    self.__injection_file = file
    self.add_var_opt('injection-file', file)

  def get_inj_file(self):
    """
    Gets the injection file
    """
    return self.__injection_file

  def set_start(self, start):
    """
    Sets GPS start time
    """
    self.__start = start

  def get_start(self):
    """
    Gets GPS start time
    """
    return self.__start

  def set_end(self, end):
    """
    Sets GPS end time
    """
    self.__end = end

  def get_end(self):
    """
    Gets GPS end time
    """
    return self.__end

  def set_ifo_tag(self,ifo_tag):
    """
    Set the ifo tag that is passed to the analysis code.
    @param ifo_tag: a string to identify one or more IFOs
    """
    self.__ifo_tag = ifo_tag

  def get_ifo_tag(self):
    """
    Returns the IFO tag string
    """
    return self.__ifo_tag

  def get_output(self):
    """
    get the name of the output file
    """
    if not self.get_ifo():
      raise InspiralError("ifos have not been set")

    fname = self.get_ifo() + "-SIRE"
    if self.get_inj_file():
      fname += "_" + self.get_inj_file().split("-")[1]
      fname += "_FOUND"

    if self.get_ifo_tag(): fname += "_" + self.get_ifo_tag()
    if self.get_user_tag(): fname += "_" + self.get_user_tag()

    if (self.get_start() and not self.get_end()) or \
        (self.get_end() and not self.get_start()):
      raise InspiralError("If one of start and end is set, both must be")

    if (self.get_start()):
      duration=self.get_end()- self.get_start()
      fname += "-" + str(self.get_start()) + "-" + str(duration)

    fname += ".xml"

    return fname

  def get_missed(self):
    """
    get the name of the missed file
    """
    if self.get_inj_file():
      return self.get_output().replace("FOUND", "MISSED")
    else:
      return None

  def finalize(self):
    """
    set the output options
    """
    output = self.get_output()

    self.add_file_opt("output", output,file_is_output_file=True)
    self.add_file_opt("summary", output.replace("xml", "txt"),file_is_output_file=True)

    if self.get_inj_file():
      self.add_file_opt('injection-file', self.get_inj_file())
      self.add_file_opt('missed-injections', self.get_missed(), file_is_output_file=True)

class CoireNode(InspiralAnalysisNode):
  """
  A CoireNode runs an instance of the inspiral coire code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_coire.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__ifos  = None
    self.__ifo_tag = None
    self.__num_slides = None
    self.__injection_file = None
    self.__output_tag = None

  def set_ifos(self, ifos):
    """
    Add the list of interferometers
    """
    self.__ifos = ifos

  def get_ifos(self):
    """
    Returns the ifos
    """
    return self.__ifos

  def set_slides(self, slides):
    """
    Add the number of time slides
    """
    self.__num_slides = slides
    self.add_var_opt('num-slides',slides)

  def get_slides(self):
    """
    Returns the number of slides
    """
    return self.__num_slides

  def set_inj_file(self, file):
    """
    Sets the injection file
    """
    if file:
      self.__injection_file = file
      self.add_var_opt('injection-file', file)

  def get_inj_file(self):
    """
    Gets the injection file
    """
    return self.__injection_file

  def set_start(self, start):
    """
    Sets GPS start time
    """
    self.__start = start

  def get_start(self):
    """
    Gets GPS start time
    """
    return self.__start

  def set_end(self, end):
    """
    Sets GPS end time
    """
    self.__end = end

  def get_end(self):
    """
    Gets GPS end time
    """
    return self.__end

  def set_ifo_tag(self,ifo_tag):
    """
    Set the ifo tag that is passed to the analysis code.
    @param ifo_tag: a string to identify one or more IFOs
    """
    self.__ifo_tag = ifo_tag

  def get_ifo_tag(self):
    """
    Returns the IFO tag string
    """
    return self.__ifo_tag

  def set_output_tag(self):
    fname = self.job().get_exec_name().upper()
    if self.get_slides(): fname += "_SLIDE"
    if self.get_inj_file():
      fname += "_" + \
          self.get_inj_file().split("/")[-1].split(".")[0].split("-")[1]
      fname += "_FOUND"
    if self.get_ifo_tag(): fname += "_" + self.get_ifo_tag()
    if self.get_user_tag(): fname += "_" + self.get_user_tag()
    self.__output_tag = fname

  def get_output_tag(self):
    return self.__output_tag

  def get_output(self):
    """
    get the name of the output file
    """
    if not self.get_ifos():
      raise InspiralError("ifos have not been set")

    self.set_output_tag()
    fname = self.get_ifos() + '-' + self.get_output_tag()

    if (self.get_start() and not self.get_end()) or \
           (self.get_end() and not self.get_start()):
      raise InspiralError("If one of start and end is set, "\
            "both must be")

    if (self.get_start()):
      duration=self.get_end() - self.get_start()
      fname += "-" + str(self.get_start()) + "-" + str(duration)

    fname += ".xml"

    return fname

  def get_missed(self):
    """
    get the name of the missed file
    """
    if self.get_inj_file():
      return self.get_output().replace("FOUND", "MISSED")
    else:
      return None

  def finalize(self):
    """
    set the output options
    """
    output = self.get_output()

    self.add_file_opt("output", output,file_is_output_file=True)
    self.add_file_opt("summary", output.replace("xml", "txt"),file_is_output_file=True)

    if self.get_inj_file():
      self.add_file_opt('injection-file', self.get_inj_file())
      self.add_file_opt('missed-injections', self.get_missed(), file_is_output_file=True)


class FrJoinNode(InspiralAnalysisNode):
  """
  A FrJoinNode runs an instance of lalapps_frjoin in a Condor DAG
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_frjoin.
    """
    InspiralAnalysisNode.__init__(self,job)

  def set_output(self, outputName):
    """
    Set the output name of the frame file
    @param outputName: name of the injection file created
    """
    self.add_var_opt('output',outputName)
    self.add_file_opt('output',outputName,file_is_output_file=True)
    self.__outputName = outputName

  def get_output(self):
    """
    Get the output name of the frame file
    """
    return self.__outputName


class CohBankNode(InspiralAnalysisNode):
  """
  A CohBankNode runs an instance of the coherent code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_coherent_inspiral.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__bank = None
    self.__ifos = None

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)
    self.__bank = bank

  def get_bank(self):
    return self.__bank

  def set_ifo(self, ifo):
    """
    Add the interferometer to the list of ifos
    ifo = IFO code (e.g. G1,L1, H1 or H2).
    """
    if ifo == 'G1':
      self.add_var_opt('g1-triggers','')
      self.__ifo_g1 = 'G1'
    elif ifo == 'H1':
      self.add_var_opt('h1-triggers','')
      self.__ifo_h1 = 'H1'
    elif ifo == 'H2':
      self.add_var_opt('h2-triggers','')
      self.__ifo_h2 = 'H2'
    elif ifo == 'L1':
      self.add_var_opt('l1-triggers','')
      self.__ifo_l1 = 'L1'
    elif ifo == 'T1':
      self.add_var_opt('t1-triggers','')
      self.__ifo_t1 = 'T1'
    elif ifo == 'V1':
      self.add_var_opt('v1-triggers','')
      self.__ifo_v1 = 'V1'

  def set_ifos(self,ifos):
    self.add_var_opt('ifos', ifos)
    self.__ifos = ifos

  def get_ifos(self):
    return self.__ifos

  def set_num_slides(self, num_slides):
    """
    Set number of time slides to undertake
    """
    self.add_var_opt('num-slides',num_slides)
    self.__num_slides = num_slides

  def get_output(self):
    """
    Returns the file name of output from the coherent bank.
    """

    if not self.get_ifos():
      raise InspiralError("Ifos have not been set")

    basename = self.get_ifos() + '-COHBANK'

    if self.get_user_tag():
      basename += '_' + self.get_user_tag()

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.get_zip_output():
      filename += '.gz'

    self.add_output_file(filename)

    return filename


class CohInspBankNode(InspiralAnalysisNode):
  """
  A CohBankNode runs an instance of the coherent code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_coherent_inspiral.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__bank = None
    self.__ifos = None

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)
    self.__bank = bank

  def get_bank(self):
    return self.__bank

  def set_ifos(self,ifos):
    self.add_var_opt('ifos', ifos)
    self.__ifos = ifos

  def get_ifos(self):
    return self.__ifos

  def set_num_slides(self, num_slides):
    """
    Set number of time slides to undertake
    """
    self.add_var_opt('num-slides',num_slides)
    self.__num_slides = num_slides

  def get_output(self):
    """
    Returns the file name of output from the coherent bank.
    """

    if not self.get_ifos():
      raise InspiralError("Ifos have not been set")

    basename = self.get_ifos() + '-COHINSPBANK'

    if self.get_user_tag():
      basename += '_' + self.get_user_tag()

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.get_zip_output():
      filename += '.gz'

    self.add_output_file(filename)

    return filename

    # overwrite standard log file names
    self.set_stdout_file('logs/' + exec_name + '-$(cluster)-$(process).out')
    self.set_stderr_file('logs/' + exec_name + '-$(cluster)-$(process).err')


class ChiaNode(InspiralAnalysisNode):
  """
  A ChiaNode runs an instance of the coherent_inspiral code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_coherent_inspiral.
    """
    InspiralAnalysisNode.__init__(self,job)

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)

  def set_ifo_tag(self,ifo_tag):
    """
    Set the ifo tag that is passed to the analysis code.
    @param ifo_tag: a string to identify one or more IFOs
    """
    self.__ifo_tag = ifo_tag

  def get_ifo_tag(self):
    """
    Returns the IFO tag string
    """
    return self.__ifo_tag

  def get_output(self):
    """
    Returns the file name of output from coherent inspiral.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo_tag():
      raise InspiralError("Start time, end time or ifos have not been set")

    basename = self.get_ifo_tag() + '-CHIA'

    if self.get_user_tag():
      basename += '_' + self.get_user_tag()

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.get_zip_output():
      filename += '.gz'

    self.add_output_file(filename)

    return filename


class CohireNode(InspiralAnalysisNode):
  """
  A CohireNode runs an instance of the inspiral cohire code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_cohire.
    """
    InspiralAnalysisNode.__init__(self,job)
    self.__ifos  = None
    self.__ifo_tag = None
    self.__num_slides = None
    self.__injection_file = None
    self.__output_tag = None

  def set_ifos(self, ifos):
    """
    Add the list of interferometers
    """
    self.__ifos = ifos

  def get_ifos(self):
    """
    Returns the ifos
    """
    return self.__ifos
  def set_slides(self, slides):
    """
    Add the number of time slides
    """
    self.__num_slides = slides
    self.add_var_opt('num-slides',slides)

  def get_slides(self):
    """
    Returns the number of slides
    """
    return self.__num_slides

  def set_inj_file(self, file):
    """
    Sets the injection file
    """
    if file:
      self.__injection_file = file
      self.add_var_opt('injection-file', file)

  def get_inj_file(self):
    """
    Gets the injection file
    """
    return self.__injection_file

  def set_start(self, start):
    """
    Sets GPS start time
    """
    self.__start = start

  def get_start(self):
    """
    Gets GPS start time
    """
    return self.__start

  def set_end(self, end):
    """
    Sets GPS end time
    """
    self.__end = end

  def get_end(self):
    """
    Gets GPS end time
    """
    return self.__end

  def set_ifo_tag(self,ifo_tag):
    """
    Set the ifo tag that is passed to the analysis code.
    @param ifo_tag: a string to identify one or more IFOs
    """
    self.__ifo_tag = ifo_tag

  def get_ifo_tag(self):
    """
    Returns the IFO tag string
    """
    return self.__ifo_tag

  def set_output_tag(self):
    fname = self.job().get_exec_name().upper()
    if self.get_slides(): fname += "_SLIDE"
    if self.get_inj_file():
      fname += "_" + \
          self.get_inj_file().split("/")[-1].split(".")[0].split("-")[1]
      fname += "_FOUND"
    if self.get_ifo_tag(): fname += "_" + self.get_ifo_tag()
    if self.get_user_tag(): fname += "_" + self.get_user_tag()
    self.__output_tag = fname

  def get_output_tag(self):
    return self.__output_tag

  def get_output(self):
    """
    get the name of the output file
    """
    if not self.get_ifos():
      raise InspiralError("ifos have not been set")

    self.set_output_tag()
    fname = self.get_ifos() + '-' + self.get_output_tag()

    if (self.get_start() and not self.get_end()) or \
           (self.get_end() and not self.get_start()):
      raise InspiralError("If one of start and end is set, "\
            "both must be")

    if (self.get_start()):
      duration=self.get_end() - self.get_start()
      fname += "-" + str(self.get_start()) + "-" + str(duration)

    fname += ".xml"

    return fname

  def get_missed(self):
    """
    get the name of the missed file
    """
    if self.get_inj_file():
      return self.get_output().replace("FOUND", "MISSED")
    else:
      return None

  def finalize(self):
    """
    set the output options
    """
    output = self.get_output()

    self.add_file_opt("output", output,file_is_output_file=True)
    self.add_file_opt("summary", output.replace("xml", "txt"),file_is_output_file=True)

    if self.get_inj_file():
      self.add_file_opt('injection-file', self.get_inj_file())
      self.add_file_opt('missed-injections', self.get_missed(), file_is_output_file=True)


class InspInjFindNode( InspiralAnalysisNode ):
  """
  An InspInjFindNode runs an instance of the InspInjJob in a
  Condor DAG.
  """
  def __init__(self, job):
    """
    @param job: A CondorDAGJob that can run an instance of ligolw_inspinjfind.
    """
    InspiralAnalysisNode.__init__(self, job)


##############################################################################
#Plotting Jobs and Nodes

class PlotInspiralrangeJob(InspiralPlottingJob):
  """
  A plotinspiralrange job. The static options are read from the section
  [plotinspiralrange] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotinspiralrange'
    sections = ['plotinspiralrange']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)

class PlotInspiralrangeNode(InspiralPlottingNode):
  """
  A PlotInspiralrangeNode runs an instance of the plotinspiral code in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotinspiralrange.
    """
    InspiralPlottingNode.__init__(self,job)

#######################################################################################

class PlotInspiralJob(InspiralPlottingJob):
  """
  A plotinspiral job. The static options are read from the section
  [plotinspiral] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotinspiral'
    sections = ['plotinspiral']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)

class PlotInspiralNode(InspiralPlottingNode):
  """
  A PlotInspiralNode runs an instance of the plotinspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotinspiral.
    """
    InspiralPlottingNode.__init__(self,job)

###########################################################################################

class PlotThincaJob(InspiralPlottingJob):
  """
  A plotthinca job. The static options are read from the section
  [plotthinca] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotthinca'
    sections = ['plotthinca']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('request_memory', '2500')

class PlotThincaNode(InspiralPlottingNode):
  """
  A PlotThincaNode runs an instance of the plotthinca code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotthinca.
    """
    InspiralPlottingNode.__init__(self,job)


###########################################################################################

class PlotCohsnrJob(InspiralPlottingJob):
  """
  A plotthinca job. The static options are read from the section
  [plotthinca] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotcohsnr'
    sections = ['plotcohsnr']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)

class PlotCohsnrNode(InspiralPlottingNode):
  """
  A PlotThincaNode runs an instance of the plotthinca code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotthinca.
    """
    InspiralPlottingNode.__init__(self,job)


#######################################################################################

class PlotNumtemplatesJob(InspiralPlottingJob):
  """
  A plotnumtemplates job. The static options are read from the section
  [plotnumtemplates] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotnumtemplates'
    sections = ['plotnumtemplates']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)

class PlotNumtemplatesNode(InspiralPlottingNode):
  """
  A PlotNumtemplatesNode runs an instance of the plotinspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotnumtemplates.
    """
    InspiralPlottingNode.__init__(self,job)

##############################################################################

class PlotEthincaJob(InspiralPlottingJob):
  """
  A plotethinca job. The static options are read from the section
  [plotethinca] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotethinca'
    sections = ['plotethinca']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('request_memory', '2500')

class PlotEthincaNode(InspiralPlottingNode):
  """
  A PlotEthincaNode runs an instance of the plotinspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotethinca.
    """
    InspiralPlottingNode.__init__(self,job)

#############################################################################

class PlotInspmissedJob(InspiralPlottingJob):
  """
  A plotinspmissed job. The static options are read from the section
  [plotinspmissed] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotinspmissed'
    sections = ['plotinspmissed']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)

class PlotInspmissedNode(InspiralPlottingNode):
  """
  A PlotInspmissedNode runs an instance of the plotinspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotinspmissed.
    """
    InspiralPlottingNode.__init__(self,job)

#############################################################################

class PlotEffdistcutJob(InspiralPlottingJob):
  """
  A ploteffdistcut job. The static options are read from the section
  [ploteffdistcut] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'ploteffdistcut'
    sections = ['ploteffdistcut']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)

class PlotEffdistcutNode(InspiralPlottingNode):
  """
  A PlotEffdistcutNode runs an instance of the
  ploteffdistcut code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of ploteffdistcut.
    """
    InspiralPlottingNode.__init__(self,job)

#############################################################################

class PlotInspinjJob(InspiralPlottingJob):
  """
  A plotinspinj job. The static options are read from the section
  [plotinspinj] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotinspinj'
    sections = ['plotinspinj']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('request_memory', '2500')

class PlotInspinjNode(InspiralPlottingNode):
  """
  A PlotInspinjNode runs an instance of the plotinspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotinspinj.
    """
    InspiralPlottingNode.__init__(self,job)

#############################################################################

class PlotSnrchiJob(InspiralPlottingJob):
  """
  A plotsnrchi job. The static options are read from the section
  [plotsnrchi] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'plotsnrchi'
    sections = ['plotsnrchi']
    extension = 'html'
    InspiralPlottingJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('request_memory', '2500')

class PlotSnrchiNode(InspiralPlottingNode):
  """
  A PlotSnrchiNode runs an instance of the plotinspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of plotsnrchi.
    """
    InspiralPlottingNode.__init__(self,job)

#############################################################################

class PlotGRBtimeslideStatsJob(InspiralAnalysisJob):
  """
  A plotgrbtimeslidestats job. The static options are read from the section
  [grbtimeslidestats] in the ini file.  The stdout and stderr from the job
  are directed to the logs directory.  The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp):
    """
    @param cp = ConfigParser object from which options are read.
    """
    exec_name = 'pylal_grbtimeslide_stats'
    sections = ['grbtimeslidestats']
    extension = 'html'
    InspiralAnalysisJob.__init__(self,cp,sections,exec_name,extension)
    self.add_condor_cmd('getenv', 'True')

class PlotGRBtimeslideStatsNode(InspiralAnalysisNode):
  """
  A PlotGRBtimeslideStatsNode runs an instance of the pylal_grbtimeslide_stats code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of pylal_grbtimeslide_stats.
    """
    InspiralAnalysisNode.__init__(self,job)

#############################################################################

class MiniFollowupsJob(InspiralPlottingJob):
  """
  A minifollowups job. Static options are read from the
  [minifollowups] section in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'minifollowups'
    sections = ['minifollowups','omega-scans']
    extension = None
    InspiralPlottingJob.__init__(self, cp, sections, exec_name, extension)
    self.add_condor_cmd('request_memory', '2500')

  def set_time_slides(self):
    """
    Turns on the --time-slides argument.
    """
    self.add_opt('time-slides', '')


class MiniFollowupsNode(InspiralPlottingNode):
  """
  A mininfollowups node.
  """
  def __init__(self, job):
    """
    @param job: a MiniFollowupsJob
    """
    InspiralAnalysisNode.__init__(self, job)
    self.__cache_file = None
    self.__cache_string = None
    self.__prefix = None
    self.__suffix = None
    self.__input_xml = None
    self.__input_xml_summary = None
    self.__output_html_table = None
    self.__table_name = None

  def set_cache_file(self, cache_file):
    """
    Set the ihope cache file to use.
    """
    self.add_file_opt( 'cache-file', cache_file )
    self.__cache_file = cache_file

  def get_cache_file(self):
    """
    Returns the cache file that's set.
    """
    return self.__cache_file

  def set_cache_string(self, cache_string):
    """
    Set the ihope cache file to use.
    """
    self.add_file_opt( 'cache-string', cache_string )
    self.__cache_string = cache_string

  def get_cache_string(self):
    """
    Returns the cache file that's set.
    """
    return self.__cache_string

  def set_prefix(self, prefix):
    """
    Sets the prefix option, which is used for plot names.
    """
    self.add_var_opt( 'prefix', prefix )
    self.__prefix = prefix

  def get_prefix(self):
    """
    Return the prefix that's set.
    """
    return self.__prefix

  def set_suffix(self, suffix):
    """
    Sets the suffix option, which is used for plot names.
    """
    self.add_var_opt( 'suffix', suffix )
    self.__suffix = suffix

  def get_suffix(self):
    """
    Return the suffix that's set.
    """
    return self.__suffix

  def set_input_xml(self, input_xml):
    """
    Sets the input xml.
    """
    self.add_var_opt( 'input-xml', input_xml)
    self.__input_xml = input_xml

  def get_input_xml(self):
    """
    Return the input_xml that's set.
    """
    return self.__input_xml

  def set_input_xml_summary(self, input_xml_summary):
    """
    Sets the input xml.
    """
    self.add_var_opt( 'input-xml-summary', input_xml_summary)
    self.__input_xml_summary = input_xml_summary

  def get_input_xml_summary(self):
    """
    Return the input_xml_summary that's set.
    """
    return self.__input_xml_summary

  def set_output_html_table(self, output_html_table):
    """
    Sets the input xml.
    """
    self.add_var_opt( 'output-html-table', output_html_table)
    self.__output_html_table = output_html_table

  def get_output_html_table(self):
    """
    Return the output_html_table that's set.
    """
    return self.__output_html_table

  def set_table_name(self, table_name):
    """
    Sets the table-name argument.
    """
    self.add_var_opt( 'table-name', table_name )
    self.__table_name = table_name

  def get_table_name(self):
    """
    Return the table_name that's set.
    """
    return self.__table_name


#############################################################################
# following are types of pipeline.SqliteJobs and Nodes

class DBSimplifyJob(pipeline.SqliteJob):
  """
  A DBSimplify job. The static options are read from the section
  [dbsimplify] in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'dbsimplify'
    sections = ['dbsimplify']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class DBSimplifyNode(pipeline.SqliteNode):
  """
  A DBSimplify node.
  """
  def __init__(self, job):
    """
    @param job: a DBSimplifyJob
    """
    pipeline.SqliteNode.__init__(self, job)


class ComputeDurationsJob(pipeline.SqliteJob):
  """
  A ComputeDurations job. The static options are read from the section
  [compute_durations] in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'compute_durations'
    sections = ['compute_durations']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class ComputeDurationsNode(pipeline.SqliteNode):
  """
  A ComputeDurations node.
  """
  def __init__(self, job):
    """
    @param job: a ComputeDurationsJob
    """
    pipeline.SqliteNode.__init__(self, job)


class DBAddInjJob(pipeline.SqliteJob):
  """
  A DBAddInj job. The static options are read from the section
  [dbaddinj] in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'dbaddinj'
    sections = ['dbaddinj']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class DBAddInjNode(pipeline.SqliteNode):
  """
  A DBAddInj node.
  """
  def __init__(self, job ):
    """
    @param job: a DBAddInj job
    """
    pipeline.SqliteNode.__init__(self, job)
    self.__injection_file = None
    self.__inj_tag = None

  def set_injection_file( self, injection_file ):
    """
    @param injection_file: Injection file for dbaddinj to
    add to the database.
    """
    self.add_file_opt( 'injection-file', injection_file )
    self.__injection_file = injection_file

  def get_injection_file( self ):
    """
    Returns injection file for this node.
    """
    return self._injection_file

  def set_inj_tag( self, inj_tag):
    """
    @param inj_tag: Injection tag used to name the injection files
    """
    self.add_var_opt( 'sim-tag', inj_tag )
    self.__inj_tag = inj_tag

  def get_inj_tag( self):
    """
    Returns injection_tag for this node.
    """
    return self.__inj_tag

class RepopCoincJob(pipeline.SqliteJob):
  """
  A repop_coinc job. The static options are read from the section
  [repop_coinc] in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'repop_coinc'
    sections = ['repop_coinc']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class RepopCoincNode(pipeline.SqliteNode):
  """
  A repop_coinc node.
  """
  def __init__(self, job):
    """
    @param job: a RepopCoincJob
    """
    pipeline.SqliteNode.__init__(self, job)


class DBInjFindJob(pipeline.SqliteJob):
  """
  A dbinjfind job. The static options are read from the section
  [dbinjfind] in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'dbinjfind'
    sections = ['dbinjfind']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class DBInjFindNode(pipeline.SqliteNode):
  """
  A dbinjfind node.
  """
  def __init__(self, job):
    """
    @param job: a DBInjFindJob
    """
    pipeline.SqliteNode.__init__(self, job)


class ClusterCoincsJob(pipeline.SqliteJob):
  """
  A cluster coincs job. The static options are read from the section
  [cluster_coincs] in the ini file.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'cluster_coincs'
    sections = ['cluster_coincs']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class ClusterCoincsNode(pipeline.SqliteNode):
  """
  A ClusterCoincs node.
  """
  def __init__(self, job):
    """
    @param job: a ClusterCoincsJob
    """
    pipeline.SqliteNode.__init__(self, job)


class CFarJob(pipeline.SqliteJob):
  """
  A cfar job. The static options are read from the section [cfar] in
  the ini file.
  """
  def __init__(self, cp, sections):
    """
    @param cp: ConfigParser object from which options are read.
    @param sections: list of sections for cp to read from
    """
    exec_name = 'cfar'
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class CFarNode(pipeline.SqliteNode):
  """
  A CFar node.
  """
  def __init__(self, job):
    """
    @param job: a CFarJob
    """
    pipeline.SqliteNode.__init__(self, job)


class LigolwCBCPrintJob(pipeline.SqliteJob):
  """
  A LigolwCBCPrintJob is a generic job class for ligolw_cbc_print* programs, e.g., ligolw_cbc_printlc.
  """
  def __init__(self, cp, exec_name, sections):
    """
    @param cp: ConfigParser object from which options are read.
    @param exec_name UNDOCUMENTED
    @param sections: list of sections for cp to read from
    """
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class LigolwCBCPrintNode(pipeline.SqliteNode):
  """
  A LigolwCBCPrintJob is a generic node class for ligolw_cbc_print* programs, e.g., ligolw_cbc_printlc.
  This class offers options common to these programs.
  """
  def __init__(self, job):
    """
    @param job: a PrintLCJob
    """
    pipeline.SqliteNode.__init__(self, job)
    self.__extract_to_xml = None
    self.__extract_to_database = None
    self.__exclude_coincs = None
    self.__include_only_coincs = None
    self.__sim_tag = None
    self.__output_format = None
    self.__columns = None

  def set_extract_to_xml(self, xml_filename):
    """
    Sets the extract-to-xml option.
    """
    self.add_var_opt('extract-to-xml', xml_filename)
    self.__extract_to_xml = xml_filename

  def get_extract_to_xml(self):
    """
    Gets xml-filename if extract-to-xml is set.
    """
    return self.__extract_to_xml

  def set_extract_to_database(self, database_filename):
    """
    Sets the extract-to-database option.
    """
    self.add_var_opt('extract-to-database', database_filename)
    self.__extract_to_database = database_filename

  def get_extract_to_database(self):
    """
    Gets database-filename if extract-to-database is set.
    """
    return self.__extract_to_database

  def set_exclude_coincs(self, exclude_coincs):
    """
    Sets exclude-coincs option.
    """
    self.add_var_opt('exclude-coincs', exclude_coincs)
    self.__exclude_coincs = exclude_coincs

  def get_exclude_coincs(self):
    """
    Gets exclude-coincs option.
    """
    return self.__exclude_coincs

  def set_include_only_coincs(self, include_only_coincs):
    """
    Sets include-only-coincs option.
    """
    self.add_var_opt('include-only-coincs', include_only_coincs)
    self.__include_only_coincs = include_only_coincs

  def get_include_only_coincs(self):
    """
    Gets include-only-coincs option.
    """
    return self.__include_only_coincs

  def set_sim_tag(self, sim_tag):
    """
    Sets the --sim-tag option.
    """
    self.add_var_opt('sim-tag', sim_tag)
    self.__sim_tag = sim_tag

  def get_sim_tag(self):
    """
    Gets sim-tag option.
    """
    return self.__sim_tag

  def set_output_format(self, output_format):
    """
    Sets the output-format option. (Note that the default
    for all ligolw_cbc_print* jobs is xml.)
    """
    self.add_var_opt('output-format', output_format)
    self.__output_format = output_format

  def get_output_format(self):
    """
    Gets the output-format option.
    """
    return self.__output_format

  def set_columns(self, columns):
    """
    Sets the columns option.
    """
    self.add_var_opt('columns', columns)
    self.__columns = columns

  def get_columns(self):
    """
    Gets the columns option.
    """
    return self.__columns


class PrintLCNode(LigolwCBCPrintNode):
  """
  A special instance of LigolwCBCPrintNode that adds printlc-specific methods.
  """
  def __init__(self, job):
    """
    @param job: a LigolwCBCPrintJob
    """
    LigolwCBCPrintNode.__init__(self, job)
    self.__datatype = None

  def set_datatype(self, datatype):
    """
    Sets datatype option.
    """
    self.add_var_opt('datatype', datatype)
    self.__datatype = datatype

  def get_datatype(self):
    """
    Gets datatype.
    """
    return self.__datatype

class PrintSimsNode(LigolwCBCPrintNode):
  """
  A special instance of LigolwCBCPrintNode that adds printsims-specific methods.
  """
  def __init__(self, job):
    """
    @param job: a LigolwCBCPrintJob
    """
    LigolwCBCPrintNode.__init__(self, job)
    self.__comparison_datatype = None
    self.__simulation_table = None
    self.__recovery_table = None

  def set_comparison_datatype(self, datatype):
    """
    Sets comparison-datatype option.
    """
    self.add_var_opt('comparison-datatype', datatype)
    self.__comparison_datatype = datatype

  def get_comparison_datatype(self):
    """
    Gets comparison-datatype.
    """
    return self.__comparison_datatype


class PrintMissedNode(LigolwCBCPrintNode):
  """
  A special instance of LigolwCBCPrintNode that adds printmissed-specific methods.
  """
  def __init__(self, job):
    """
    @param job: a LigolwCBCPrintJob
    """
    LigolwCBCPrintNode.__init__(self, job)


class PlotSlidesJob(pipeline.SqliteJob):
  """
  A plotslides job. The static options are read from the sections [plot_input]
  and [plotslides].
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'plotslides'
    sections = ['plot_input', 'plotslides']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)

  def set_plot_playground_only(self):
    """
    Sets plot-playground-only option. This causes job to only plot playground.
    """
    self.add_opt('plot-playground-only','')


class PlotSlidesNode(pipeline.SqliteNode):
  """
  A PlotSlides node.
  """
  def __init__(self, job):
    """
    @param job: a PlotSlidesJob
    """
    pipeline.SqliteNode.__init__(self, job)


class PlotCumhistJob(pipeline.SqliteJob):
  """
  A plotcumhist job. The static options are read from the sections [plot_input] and
  [plotcumhist].
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'plotcumhist'
    sections = ['plot_input', 'plotcumhist']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)

  def set_plot_playground_only(self):
    """
    Sets plot-playground-only option. This causes job to only plot playground.
    """
    self.add_opt('plot-playground-only','')


class PlotCumhistNode(pipeline.SqliteNode):
  """
  A PlotCumhist node.
  """
  def __init__(self, job):
    """
    @param job: a PlotCumhist Job
    """
    pipeline.SqliteNode.__init__(self, job)


class PlotIfarJob(pipeline.SqliteJob):
  """
  A plotifar job. The static options are read from the [plotifar] section.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'plotifar'
    sections = ['plot_input','plotifar']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)


class PlotIfarNode(pipeline.SqliteNode):
  """
  A PlotIfar node.
  """
  def __init__(self, job):
    """
    @param job: a PlotIfarJob
    """
    pipeline.SqliteNode.__init__(self, job)
    self.__datatype = None

  def set_datatype(self, datatype):
    """
    Sets datatype option.
    """
    self.add_var_opt('datatype', datatype)
    self.__datatype = datatype

  def get_datatype(self):
    """
    Gets datatype.
    """
    return self.__datatype

class PlotFMJob(pipeline.SqliteJob):
  """
  A plotfm job. The static options are read from the [plotfm] seciont.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which objects are read.
    """
    exec_name = 'plotfm'
    sections = ['plot_input', 'plotfm']
    pipeline.SqliteJob.__init__(self, cp, sections, exec_name)

class PlotFMNode(pipeline.SqliteNode):
  """
  A PlotFM node.
  """
  def __init__(self, job):
    """
    @param job: a PlotFMJob
    """
    pipeline.SqliteNode.__init__(self, job)
    self.__sim_tag = None

  def set_sim_tag(self, sim_tag):
    """
    Sets the --sim-tag option.
    """
    self.add_var_opt('sim-tag', sim_tag)
    self.__sim_tag = sim_tag

  def get_sim_tag(self):
    """
    Gets sim-tag option.
    """
    return self.__sim_tag

##############################################################################
# some functions to make life easier later

def overlap_test(interval1, interval2, slide_sec=0):
  """
  Test whether the two intervals could possibly overlap with one of them being
  slid by a maximum time of slide_sec.  Perform three tests:
  1)  Does the start of interval 1 lie within interval 2's range (with the
    start decremented by slide_sec and the end incremented by slide_sec)
  2)  Does the end of interval 1 lie within interval 2's range (with the start
    decremented by slide_sec and the end incremented by slide_sec)
  3)  Does interval 1 completely cover (the extended) interval 2,
    ie is interval_1 start before (interval 2 start - slide_sec) AND
    interval 1 end after (interval 2 end + slide_sec)
  If any of the above conditions are satisfied then return True, else False.
  """
  start1 = interval1.start()
  end1 = interval1.end()
  left = interval2.start() - slide_sec
  right = interval2.end() + slide_sec

  return (start1 >= left and start1 <= right) or \
         (end1 >= left and end1 <= right) or \
         (start1 <= left and end1 >= right)


class SearchVolumeJob(pipeline.SqliteJob):
  """
  A search volume job. Computes the observed physical volume
  above a specified FAR; if FAR is not specified, computes the
  volume above the loudest event (open box) or FAR=1/livetime
  (closed box).
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = 'search_volume'
    pipeline.SqliteJob.__init__(self, cp, ['search-volume'], exec_name)
    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

class SearchVolumeNode(pipeline.SqliteNode):
  """
  A search volume node.
  """
  def __init__(self, job):
    """
    """
    pipeline.SqliteNode.__init__(self, job)

  def add_database(self, db):
    self.add_var_arg(db)

  def set_output_cache(self, file):
    self.add_var_opt("output-cache", file)

  def set_user_tag(self, tag):
    self.add_var_opt("user-tag", tag)

  def set_veto_segments_name(self, name):
    self.add_var_opt("veto-segments-name", name)

  def set_open_box(self):
    self.add_var_arg("--open-box")


class SearchUpperLimitJob(pipeline.SqliteJob):
  """
  A search upper limit job. Compute the search upper limit from the search
  volume output. Generates upper limit plots.
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    sections: list of sections for cp to read from
    """
    exec_name = 'search_upper_limit'
    pipeline.SqliteJob.__init__(self, cp, ['upper-limit'], exec_name)
    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

class SearchUpperLimitNode(pipeline.SqliteNode):
  """
  A search upper limit node.
  """
  def __init__(self, job):
    """
    @param job: a SearchUpperLimitJob
    """
    pipeline.SqliteNode.__init__(self, job)
    self.open_box = False

  def add_input_cache(self, input_cache):
    self.add_var_arg(input_cache)

  def set_user_tag(self, tag):
    self.add_var_opt("user-tag", tag)

  def set_open_box(self):
    '''
    Set the open box flag.
    '''
    if not self.open_box:
      self.open_box = True
      self.add_var_arg("--open-box")

class MVSCDagGenerationJob(InspiralAnalysisJob):
  """
  a job that generates the mvsc_dag, which will be run as an external subdag
  """
  def __init__(self, cp):
    """
    @param cp: ConfigParser object from which options are read.
    """
    exec_name = "mvsc_dag"
    universe = "vanilla"
    sections = "[mvsc_dag]"
    executable = cp.get('condor',exec_name)
    pipeline.CondorDAGJob.__init__(self, universe, executable)
    pipeline.AnalysisJob.__init__(self, cp)
    self.add_condor_cmd('getenv','True')
    self.set_stdout_file('logs/' + exec_name + '-$(cluster)-$(process).out')
    self.set_stderr_file('logs/' + exec_name + '-$(cluster)-$(process).err')
    self.set_sub_file(exec_name + '.sub')

class MVSCDagGenerationNode(InspiralAnalysisNode):
  """
  the node that runs the mvsc dag generation script for a given configuration
  generally the different nodes will be for different categories of vetoes
  """
  def __init__(self, job):
    InspiralAnalysisNode.__init__(self, job)
  def set_database(self, database):
    self.add_var_arg(database)
  def set_user_tag(self, tag):
    self.add_var_opt("user-tag",tag)
class ExtendedCoincJob(InspiralAnalysisJob):
  """
  job to calculate the extende background for zero far events
  """
  def __init__(self, cp):
    """
    cp = ConfigParser object from which options are read.
    sections = sections of the ConfigParser that get added to the opts
    exec_name = exec_name name in ConfigParser
    """

    exec_name = 'extended_background'
    sections = []
    extension = 'html'
    InspiralAnalysisJob.__init__(self, cp, sections, exec_name, extension)
    self.add_condor_cmd('getenv','True')


class ExtendedCoincNode(InspiralAnalysisNode):
  """
  Node to calculate the extended background for a zero far event
  """
  def __init__(self, job):
    InspiralAnalysisNode.__init__(self, job)

  def set_coinc_threshold(self, coinc_threshold):
        self.add_var_opt('coinc-threshold', coinc_threshold)

  def set_ihope_base_dir(self, base_dir):
        self.add_var_opt('ihope-base-dir', base_dir)

  def set_param_ranges(self, param_ranges):
        self.add_var_opt('param-ranges', param_ranges)

  def set_ethinca(self, ethinca):
        self.add_var_opt('e-thinca-parameter', ethinca)

  def set_slide_step(self, slide_step):
        self.add_var_opt('slide-step', slide_step)

  def set_veto_window(self, veto_window):
        self.add_var_opt('veto-window', veto_window)

  def set_new_snr_cut(self, new_snr_cut):
        self.add_var_opt('new-snr-cut', new_snr_cut)

  def set_loudest_event_glob(self, event_glob):
        self.add_var_opt('loudest-event-glob', event_glob)
