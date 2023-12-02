# Copyright (C) 2008-2013 Bernd Feige
# This file is part of avg_q and released under the GPL v3 (see avg_q/COPYING).
import os
from .avg_q import escape_filename

# Format, list_of_extensions
# Extensions are first matched case sensitive, then lowercased.
formats_and_extensions=[
 ('NeuroScan', ['.avg','.eeg','.cnt']),
 ('BrainVision', ['.vhdr','.ahdr','.vmrk','.amrk','.eeg']),
 ('asc', ['.asc']),
 ('hdf', ['.hdf']),
 ('edf', ['.edf','.rec','.bdf']),
 ('freiburg', ['.co']),
 ('neurofile', ['.eeg']),
 ('nirs', ['.nirs']),
 ('nke', ['.eeg', '.EEG']),
 ('Inomed', ['.emg','.trg']),
 ('sound', ['.wav','.au','.snd','.aiff','.mp3','.ogg']),
 ('Coherence', ['.Eeg','.EEG', '.eeg']),
 ('Konstanz', ['.sum', '.raw']),
 ('Vitaport', ['.vpd', '.raw']),
 ('Tucker', ['.raw']),
 ('Embla', ['.ebm']),
 ('Unisens', ['.bin','.csv']),
 ('CFS', ['.cfs']),
 ('Sigma', ['.EEG']),
]

class avg_q_file(object):
 def __init__(self,filename=None,fileformat=None):
  if filename:
   # cf. https://github.com/pandas-dev/pandas/blob/325dd686de1589c17731cf93b649ed5ccb5a99b4/pandas/io/common.py#L131-L160
   if not isinstance(filename, str):
    if hasattr(filename, '__fspath__'):
     filename=filename.__fspath__()
    else:
     filename=str(filename)
   if not fileformat:
    filename,fileformat=self.guessformat(filename)
  self.filename=filename
  self.fileformat=fileformat
  self.epoched=False
  self.addmethods=None
  self.getepochmethod=None
  self.trigfile=None
  if fileformat is None:
   self.getepochmethod=None
  elif fileformat=='BrainVision':
   self.getepochmethod='''
read_brainvision %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='NeuroScan':
   self.getepochmethod='''
read_synamps %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
   name,ext=os.path.splitext(filename)
   if ext.lower() in ['.avg','.eeg']:
    self.epoched=True
  elif fileformat=='asc':
   self.getepochmethod='''
readasc %(fromepoch_arg)s %(epochs_arg)s %(filename)s
'''
   self.epoched=True
  elif fileformat=='hdf':
   self.getepochmethod='''
read_hdf %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='edf':
   self.getepochmethod='''
read_rec %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='freiburg':
   if os.path.exists(self.filename):
    # Remove trailing .co - see documentation of read_freiburg, which needs
    # only the name without extension to read an SL .co + .coa combination
    if self.filename.lower().endswith('.co'):
     self.filename=self.filename[:-3]
   self.getepochmethod='''
read_freiburg %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(filename)s %(aftertrig)s
'''
  elif fileformat=='Vitaport':
   self.getepochmethod='''
read_vitaport %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='neurofile':
   self.getepochmethod='''
read_neurofile %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='nirs':
   from . import nirs
   '''
   NOTE Special case for files to be read using an Epochsource such as numpy_Epochsource
   This is handled specially in avg_q.Epochsource()
   '''
   self.getepochmethod=nirs.nirs_Epochsource(self.filename)
  elif fileformat=='nke':
   self.getepochmethod='''
read_nke %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='Inomed':
   self.getepochmethod='''
read_inomed %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='sound':
   self.getepochmethod='''
read_sound %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='Coherence':
   from . import Coherence
   coherencefile=Coherence.avg_q_Coherencefile(filename)
   self.getepochmethod=coherencefile.getepochmethod
  elif fileformat=='Embla':
   from . import Embla
   emblafile=Embla.avg_q_Emblafile(filename)
   self.getepochmethod=emblafile.getepochmethod
  elif fileformat=='Unisens':
   from . import Unisens
   Unisensfile=Unisens.avg_q_Unisensfile(filename)
   self.getepochmethod=Unisensfile.getepochmethod
  elif fileformat=='Konstanz':
   self.getepochmethod='''
read_kn %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(filename)s
'''
  elif fileformat=='Tucker':
   self.getepochmethod='''
read_tucker %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='CFS':
   self.getepochmethod='''
read_cfs %(fromepoch_arg)s %(epochs_arg)s %(filename)s
'''
  elif fileformat=='Sigma':
   self.getepochmethod='''
read_sigma %(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s
'''
  elif fileformat=='generic':
   # Handled specially because read_generic reads headerless data
   # and meta info must be given as options.
   self.getepochmethod=None
   # read_generic_options and read_generic_data_type can/must be set
   # accordingly before calling getepoch().
   # read_generic_options can contain any non-standard options of read_generic,
   # e.g. '-s 200 -x xchannelname -O 1' but *not* options from the standard
   # set such as -c, -f, -e, -t, -T etc. which are handled by getepoch.
   self.read_generic_options=''
   self.read_generic_data_type='string'
  elif fileformat in ['dip_simulate', 'null_source']:
   # Handled specially
   self.getepochmethod=None
  else:
   raise Exception("Unknown fileformat %s" % fileformat)
 def __str__(self):
  return self.filename+': '+self.fileformat
 def getepoch(self, beforetrig=0, aftertrig=0, continuous=False, fromepoch=None, epochs=None, offset=None, triglist=None, trigfile=None, trigtransfer=False):
  '''Construct a get_epoch line using the filetype-specific template and the
  parameters. We allow self.trigfile to be the default trigfile if set; this
  eases the construction e.g. of trigger transfer without having to pass
  trigfile for every getepoch call.'''
  if not trigfile and self.trigfile:
   trigfile=self.trigfile
  if self.fileformat=='dip_simulate':
   return '''
dip_simulate 100 %(epochs_arg)s %(beforetrig)s %(aftertrig)s eg_source
''' % {
    'epochs_arg': str(epochs),
    'beforetrig': str(beforetrig),
    'aftertrig': str(aftertrig)
   }
  elif self.fileformat=='null_source':
   return '''
null_source 100 %(epochs_arg)s 32 %(beforetrig)s %(aftertrig)s
''' % {
    'epochs_arg': str(epochs),
    'beforetrig': str(beforetrig),
    'aftertrig': str(aftertrig)
   }
  elif self.fileformat=='generic':
   self.getepochmethod='''
read_generic %(read_generic_options)s %(std_args)s %(read_generic_data_type)s
''' % {
    'read_generic_options': str(self.read_generic_options),
    'std_args': '%(continuous_arg)s %(fromepoch_arg)s %(epochs_arg)s %(offset_arg)s %(triglist_arg)s %(trigfile_arg)s %(trigtransfer_arg)s %(filename)s %(beforetrig)s %(aftertrig)s',
    'read_generic_data_type': str(self.read_generic_data_type),
   }
  return self.getepochmethod % {
   'continuous_arg': '-c' if continuous else '',
   'fromepoch_arg': '-f %d' % fromepoch if fromepoch is not None else '',
   'epochs_arg': '-e %d' % epochs if epochs is not None else '',
   'offset_arg': '-o %s' % offset if offset is not None else '',
   'triglist_arg': '-t %s' % triglist if triglist is not None else '',
   'trigfile_arg': '-R %s' % escape_filename(trigfile) if trigfile is not None else '',
   'trigtransfer_arg': '-T' if trigtransfer else '',
   'filename': escape_filename(self.filename),
   'beforetrig': str(beforetrig),
   'aftertrig': str(aftertrig)
  } + ((self.addmethods+'\n') if self.addmethods else '')
 def guessformat(self,filename):
  name,ext=os.path.splitext(filename)
  format_and_score=[]
  for format,extlist in formats_and_extensions:
   score=0
   if ext in extlist:
    score=5
   lext=ext.lower()
   if score==0 and lext in [x.lower() for x in extlist]:
    score=1
   if lext=='.eeg' and score>0:
    # Check for additional conditions; Presence increases, absence decreases the score
    # relative to formats matching only the extension (5)
    if format in ['nke', 'BrainVision', 'Coherence']:
     if format=='nke' and (os.path.exists(name+'.21e') or os.path.exists(name+'.21E')) or\
        format=='BrainVision' and os.path.exists(name+'.vhdr') or\
        format=='Coherence' and (ext=='.Eeg' or name[:-1].endswith('_000')):
      score+=2
     else:
      score-=2
   if score>0:
    format_and_score.append((format,score))
  if len(format_and_score)==0:
   raise Exception("Can't guess format of %s!" % filename)
  format_and_score.sort(key=lambda x: x[1],reverse=True)
  return filename,format_and_score[0][0]
