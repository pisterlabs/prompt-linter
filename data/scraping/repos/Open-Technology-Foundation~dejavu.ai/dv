#!/usr/bin/python3
"""
DéjàVu GPT Terminal Chatbot and Scripting
Gary Dean garydean@yatti.id
https://github.com/Open-Technology-Foundation/dejavu.ai

#: git clone https://github.com/Open-Technology-Foundation/dejavu.ai /tmp/dejavu \
    && /tmp/dejavu/dv.install
"""
# pylint: disable=global-statement
# pylint: disable=wildcard-import
# pylint: disable=line-too-long
# pylint: disable=wrong-import-position
# pylint: disable=invalid-name
# pylint: disable=broad-exception-caught
# pylint: disable=multiple-statements

import os
import datetime
import pytz
import time
import random
import re
import readline
import signal
import subprocess
import sys
import textwrap
import openai
import tiktoken
from colorama import Fore, Style

"""
Script name and directory is derived from actual basename of this script (argv0).
"""
ScriptName  = os.path.realpath(sys.argv.pop(0))
ScriptDir   = os.path.dirname(ScriptName)
ScriptName  = os.path.basename(ScriptName)
""" If the script direcory is not present in the current PATH, then append it. """
if ScriptDir not in sys.path: sys.path.append(ScriptDir)
from dejavu_std import *
from awesome_prompts import *

""" Raw stdout Output flag; True == no color, default is False. """
RawOutput = False

""" The local user home directory is HOME+/.+ScriptName. It is created unconditionally.  """
dvHome = f'{HOME}/.{ScriptName}'
os.makedirs(dvHome, exist_ok=True)
""" If the file dvHome/default.dv does not exist, then use this as a flag 
that the directory is empty and copy all example .dv scripts to this directory. """
if not os.path.exists(dvHome + '/default.dv'):
  printinfo(f'Welcome to DéjàVu, {USER.title()}.')
  if not copy_files_recursive(ScriptDir, dvHome, '*.dv', verbose=True):
    printerr('Default files copy error.')
    sys.exit()
  string = readfile(dvHome + '/default.dv')
  string = re.sub(r'\n!USER_NAME \S+', '\n!USER_NAME ' + USER.upper(), string)
  string = re.sub(r'\n/USER_NAME \S+', '\n/USER_NAME ' + USER.upper(), string)
  writefile(dvHome + '/default.dv', string)

""" Define cannonical repository URLs for dejavu """
REPOSITORY = 'https://github.com/Open-Technology-Foundation/dejavu.ai.git'
REPOSITORY_VERSION = f'https://raw.githubusercontent.com/Open-Technology-Foundation/dejavu.ai/master/dejavu.version?xyz=420{int(random.random()*1000000)}'
""" Dejavu Version Check Flag, default False.  When set, this flag will trigger an install/upgrade process from the above repository URL. """
UpdateCheck = False

# Process command line and define ConvFile to use
Version = readfile(f'{ScriptDir}/dejavu.version').strip()

# ----------------------------------------------------------------------------
def find_dv_file(dvfilename: str, **kwargs) -> str:
  """
  Finds/Creates fqfn for filenames with extension .dv.
  If not mustexist and file does not exist
    then defaults to sys.path[0]+'/'+filename.ext
    and creates file with contents of /usr/share/dejavu.ai/default.dv.

  Return '' if fail.
  """
  mustexist = kwargs.get('mustexist', True)
  searchpaths = kwargs.get('searchpaths', [dvHome, './', HOME])
  dvfilename = find_file(dvfilename, mustexist=mustexist, searchpaths=searchpaths, ext='.dv')
  if len(dvfilename) == 0:
    return ''
  if not os.path.exists(dvfilename):
    try:
      dvtext = readfile(ScriptDir + '/default.dv')
      dvtext = re.sub(r'\n!USER_NAME \S+', '\n!USER_NAME ' + USER.upper(), dvtext)
      dvtext = re.sub(r'\n/USER_NAME \S+', '\n/USER_NAME ' + USER.upper(), dvtext)
      writefile(dvfilename, dvtext)
    except:
      printerr('Could not create script ' + dvfilename)
      return ''
    printinfo('New dv script ' + dvfilename + ' created.')
  return dvfilename


ConvFile      = ''
cmdTypeAhead  = []
argvTypeAhead = []
Instructions  = []
cmdEcho       = True
Verbose       = True
cmdExit       = False
AutoSave      = False

# ----------------------------------------------------------------------------
def read_dvfile(dvfile: str) -> bool:
  """  Read a .dv script file. """
  global cmdTypeAhead, argvTypeAhead, Instructions, SYSTEM_NAME
  cmdTypeAhead = []; Instructions = []
  dvfile = find_dv_file(dvfile, mustexist=True, ext='.dv')
  if len(dvfile) == 0:
    printerr(f'Script "{dvfile}" does not exist.')
    return False
  lne = readfile(dvfile)
  Lines = lne.split('\n')
  lne   = ''
  while len(Lines) > 0:
    line = Lines.pop(0).rstrip()
    if not line:
      Instructions.append(''); continue
    if line[0] == '#':
      Instructions.append(line); continue
    # handle \ line continuations
    while line[-1] == '\\' and len(Lines) > 0:
      line = line[0:-1] + Lines.pop(0).rstrip()
    lne = line.rstrip('\r\n')
    if not lne: lne = ''; continue
    if lne[0] == '!' or lne[0] == '/':
      alne = lne.split()
      if (alne[0] == '/prompt' or alne[0] == '!prompt') and len(alne) > 1:
        if alne[1] != '"""':
          Instructions.append('/prompt ' + ' '.join(alne[1:]))
          lne = ''
          continue
        promptstr = ''
        while len(Lines) > 0:
          line = Lines.pop(0).rstrip()
          if line == '"""':
            Instructions.append('/prompt ' + promptstr)
            break
          promptstr += line + '\\n'
      elif alne[0] == '/instruction' or alne[0] == '!instruction' and len(alne) > 1:
        if alne[1] != '"""':
          Instructions.append('/instruction ' + ' '.join(alne[1:]))
          lne = ''
          continue
        instr = ''
        while len(Lines) > 0:
          line = Lines.pop(0).rstrip()
          if line == '"""':
            Instructions.append('/instruction ' + instr)
            break
          instr += line + '\\n'
      else:
        Instructions.append(' '.join(alne))
      lne = ''
      continue
    Instructions.append('/instruction ' + lne)
    lne = ''
  # end loop processing
  lne = lne.rstrip('\r\n')
  if lne: Instructions.append('/instruction ' + lne)
  cmdTypeAhead = Instructions.copy()
  cmdTypeAhead = [x for x in cmdTypeAhead if not x.startswith('#') and x]
  cmdTypeAhead.insert(0, '/echo off')
  if Verbose:
    cmdTypeAhead.append('/status short')
    cmdTypeAhead.append('/echo on')
  if len(argvTypeAhead) > 0:
    for lne in argvTypeAhead: cmdTypeAhead.append(lne)
    argvTypeAhead = []
  SYSTEM_NAME = os.path.splitext(os.path.basename(dvfile))[0].replace(' ', '_').upper()
  return True

# ----------------------------------------------------------------------------
def dvUsage():
  """ dv short invocation help """
  print('DéjàVu - GPT Terminal vs ' + Version)
  print("Usage: " + ScriptName + f""" [-vqlfxuV] [-c cmd] [dvfile]
Where 'dvfile' is a DéjàVu script file.dv. 
Defaults to '~/.dv/default.dv'
 -l|--list      List all DéjàVu scripts in '~/.dv/'.
 -a|--autosave on|off
                If on, upon exit, append current conversation to current dv script. Default is off.
 -c|--cmd 'cmd' Execute 'cmd' on entry to DéjàVu.
 -x|--exit      Immediately exit DéjàVu after first command has been executed.
 -v|--verbose   Verbose on. Default is on.
 -q|--quiet     Verbose off.
 -V|--version   Print DéjàVu version.
 -C|--no-color  Do not use color. Default is to use color.
 -u|--upgrade   Upgrade DéjàVu from git repository.
                Git repository is set to:
                  {REPOSITORY}
 --help         Full Help manpages.
""")

# argv processing ------------------------------------------------------------
updateDV = False
while len(sys.argv) > 0:
  sysargv = sys.argv.pop(0).strip()
  if not sysargv: continue
  if sysargv in ['-u', '--upgrade', '--update']:
    updateDV = True
  elif sysargv in ['-V', '--version']:
    print(f'{ScriptName} vs {Version}')
    sys.exit(0)
  elif sysargv in ['-v', '--verbose']:
    Verbose = 1; cmdEcho = 1
  elif sysargv in ['-q', '--quiet']:
    Verbose = 0; cmdEcho = 0
  elif sysargv in ['-l', '--list']:
    printinfo(f'Dejavu script files in {dvHome}')
    for listfile in os.listdir(dvHome):
      if listfile.endswith('.dv'):
        filesize = os.stat(f'{dvHome}/{listfile}').st_size / 1024
        modified_date = datetime.datetime.fromtimestamp(os.stat(dvHome + '/' + listfile).st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        printstd(f'{listfile:12s} {filesize:7.2f}KB {modified_date:20s}')
    sys.exit(0)
  elif sysargv in ['-c', '--cmd']:
    if len(sys.argv) > 0:
      argvparam = sys.argv.pop(0).strip()
      if argvparam[0] == '\\': argvparam = argvparam[1:]
      argvTypeAhead.append(argvparam)
    else:
      printerr('Command was not specified for ' + sysargv)
      sys.exit(1)
  elif sysargv in ['-a', '--autosave']:
    AutoSave = True
  elif sysargv in ['-x', '--exit']:
    Verbose = False
    cmdExit = True
  elif sysargv in ['-C', '--no-color']:
    useColor = False
    RawOutput=True
  elif sysargv in ['-h', '-?']:
    dvUsage()
    sys.exit()
  elif sysargv == '--help':
    os.execvp('man', ['man', ScriptName])
    sys.exit()
  # de-aggregate aggregated short options
  elif re.match(r'^-[caCfvqVhuxl]', sysargv):
    sys.argv = [''] + [f'-{c}' for c in sysargv[1:]] + sys.argv
  elif re.match(r'^--', sysargv):
    printerr('Invalid option ' + sysargv)
    sys.exit(1)
  elif sysargv[0:1] == '-':
    printerr('Invalid option ' + sysargv)
    sys.exit(1)
  else:
    ConvFile = sysargv

""" 
Intercept update instruction from command line 
and update this program from the repository. 
Exit after finishing.
"""
if updateDV:
  tempdir_upgrade = tempname('upgrade', '')
  os.makedirs(tempdir_upgrade, exist_ok=True)
  os.chdir(tempdir_upgrade)
  qqq = '-v' if Verbose else '-q'
  subprocess.call(['git', 'clone', qqq, REPOSITORY, tempdir_upgrade])
  installcmd = f'{tempdir_upgrade}/{ScriptName}.install'
  installargs = [installcmd, qqq, '-a' if qqq == '-q' else '']
  try:
    os.execvp(installcmd, installargs)
  except:
    sys.exit(0) # should never get here.


# Conversation script validation and default ---------------------------------
"""
If a Conversation .dv Script has not been specified on the command line, 
then ConvFile defaults to the file default.dv in the dvHome directory.
"""
if len(ConvFile) == 0:
  ConvFile = dvHome + '/default.dv'
try:
  ConvFile = find_dv_file(ConvFile, mustexist=False)
  if len(ConvFile) == 0:
    printerr(f'DéjàVu script "{ConvFile}" could not be opened.')
    sys.exit(1)
except Exception as e:
  printerr(f'DéjàVu script "{ConvFile}" could not be created.', str(e))
  sys.exit(1)
""" The ConvFile basename. """
ConvFileName = os.path.basename(ConvFile)

# ----------------------------------------------------------------------------
def getOpenAIKeys() -> bool:
  """ # Get OpenAI API keys  """
  try:
    openai.api_key = os.environ['OPENAI_API_KEY']
  except KeyError:
    printerr('Environment variable OPENAI_API_KEY is not defined.')
    printinfo('Go to https://openai.com/api for your own API key.',
              '  $ export OPENAI_API_KEY="your_key"',
              'If you set up your openai account as an organization, you will',
              'also have to set OPENAI_ORGANIZATION_ID:',
              '  $ export OPENAI_ORGANIZATION_ID="your_organization"',
              'Both these environment variables should be set in your ~/.bashrc',
              'file or in /etc/bash.bashrc.')
    sys.exit(1)
  try:
    openai.organization = os.environ['OPENAI_ORGANIZATION_ID']
  except KeyError:
    openai.organization = ''
  return True
getOpenAIKeys()


"""
#!/usr/bin/python3
import os
import sys
class Agent:
  SYSTEM_NAME = 'SYSTEM_AGENT'
  USER_NAME   = 'AGENT_USER'
  AI_NAME     = 'AI_ASSISTANT'
  prompt      = ''
  engine      = 'gpt-3.5-turbo'
  token_limit = 4096 if (engine in [ 'text-davinci-003', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']) else 2048
  temperature = 1.0
  top_p       = 1.0
  response_tokens = int(token_limit/2)
  freq_pen    = 0.0
  pres_pen    = 0.0
  stop        = []
  def __init__(self):
    pass
  def assign(self, **kwargs):
    for key, value in kwargs.items():
      if hasattr(self, key):
        setattr(self, key, value)
      else:
        print(f'{str(key)} is not a valid attribute of the `Agent` class.', file=sys.stderr)
agent = Agent()
agent.assign(engine='text-davinci-003', temperature=0.8)
print(agent.engine) # Output: text-davinci-003
print(agent.temperature) # Output: 0.8
agent.assign(invalid_key='test')
"""
SYSTEM_NAME     = 'SYSTEM_AGENT'
USER_NAME       = 'AGENT_USER'
AI_NAME         = 'AI_ASSISTANT'
prompt          = ''
Prompt          = prompt
engine          = 'gpt-3.5-turbo'
token_limit     = 4000
response_tokens = int(token_limit/2)
temperature     = 1.0
top_p           = 1.0
freq_pen        = 0.0
pres_pen        = 0.0
stop            = ''

if not read_dvfile(ConvFile):
  printerr('Error reading ' + ConvFile)
  sys.exit(1)
historyFile = initHistory(ConvFile)

conversation = []
text_block  = ''

# ----------------------------------------------------------------------------
def num_tokens_from_string(gptstring: str, encoding_name: str='gpt2') -> int:
  """Returns the number of tokens in a text gpt string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(gptstring))
  return num_tokens


# ----------------------------------------------------------------------------
def gpt35_completion(gprompt: str, gconversation: list=[], **kwargs):
  global token_limit, used_tokens
  gengine     = kwargs.get('engine', 'gpt-3.5-turbo')
  gtemperature= float(max(0.0, min(1.0, kwargs.get('temperature', 0.7))))
  gtop_p      = float(max(0.0, min(1.0, kwargs.get('top_p', 0.9))))
  gtokens     = int(kwargs.get('tokens', -1))
  gfreq_pen   = float(max(-2.0, min(2.0, kwargs.get('freq_pen', 0.2))))
  gpres_pen   = float(max(-2.0, min(2.0, kwargs.get('pres_pen', 0.65))))
  gstop       = kwargs.get('stop', [])
  gtimeout    = int(max(2, min(99420, kwargs.get('timeout', -1))))

  gprompt = role_tag_replace(gprompt.encode(encoding='ASCII', errors='ignore').decode())
  used_tokens = num_tokens_from_string(gprompt)
  # If gtokens (aka max_tokens, or response_tokens) is <=0, 
  # then gtokens is calculated to max amount.
  if gtokens <= 0:
    gtokens = max(16, (token_limit - 200) - used_tokens)
  if (gtokens + used_tokens) > token_limit:
    printerr(f'Too many response tokens requested ({gtokens}).', 'Try reducing /tokens, or deleting or summarising some of your conversation.') 
    return ''
  if Verbose and cmdEcho:
    printinfo(f'{used_tokens} total tokens in prompt')
  printlog(f'gtokens={gtokens}, engine={engine}, conv={len(gconversation)}')
  
#  gprompt = role_tag_replace(gprompt.encode(encoding='ASCII', errors='ignore').decode())
  # chats for chatQA models have to be structured differently
  if gengine in ['gpt-3.5-turbo', 'gpt-4'] and len(gconversation) > 0:
    messages = [{"role": "system", "name": SYSTEM_NAME, "content": gprompt}]
    for conv in gconversation:
      if conv.startswith('<<AI_NAME>>:'):
        role = 'assistant'
        name = AI_NAME
        conv = conv[12:].lstrip()
      elif conv.startswith('<<USER_NAME>>:'):
        role = 'user'
        name = USER_NAME
        conv = conv[14:].lstrip()
      elif conv.startswith('<<SYSTEM>>:'):
        role = 'system'
        name = SYSTEM_NAME
        conv = conv[11:].lstrip()
      else:
        role = 'system'
        name = SYSTEM_NAME
      messages.append({"role": role, "name": name, "content": role_tag_replace(conv)})
    printlog(json.dumps(messages))
    try:
      response = openai.ChatCompletion.create(
          model=gengine,
          messages=messages,
          temperature=gtemperature,
          max_tokens=gtokens,
          top_p=gtop_p,
          frequency_penalty=gfreq_pen,
          presence_penalty=gpres_pen,
          stop=gstop,
          timeout=gtimeout)
    except Exception as gpte:
      printerr('GPT experienced an error.', str(gpte))
      return ''
    for choice in response.choices:
      if "text" in choice: return choice.text
    # If no response with text is found, return the first response's content (which may be empty)
    return response.choices[0].message.content
  else:
    try:
      gpt3_response = openai.Completion.create(
          engine=gengine,
          prompt=gprompt,
          temperature=gtemperature,
          max_tokens=gtokens,
          top_p=gtop_p,
          frequency_penalty=gfreq_pen,
          presence_penalty=gpres_pen,
          stop=gstop,
          timeout=90)
    except Exception as gpte:
      printerr('GPT experienced an error.', str(gpte))
      return ''
    return gpt3_response['choices'][0]['text'].strip('\n')


# ---------------------------------------------------------------------------
def cmd_help():
  """ dejavu command help """
  printinfo('DéjàVu System Commands', style=Style.BRIGHT)
  json_data = json.loads(readfile(ScriptDir + '/dejavu-command-help.json'))
  rowcount = 1
  for cmdhelp in json_data:
    if (rowcount + len(cmdhelp)) >= getScreenRows():
      rowcount = 0
      if is_terminal(sys.stdout):
        try:
          input(Style.RESET_ALL + 'More...')
          print('\r\x1b[2K', end='')
        except KeyboardInterrupt:
          print('^C', file=sys.stderr)
          break
    if len(cmdhelp) == 1:
      printinfo(cmdhelp[0], style=Style.BRIGHT)
      rowcount += 1
      continue
    printinfo(f' {cmdhelp[0]:17s} {cmdhelp[1]}')
    rowcount += 1
    for chelp in cmdhelp[2:]:
      printinfo(f' %17s {chelp}' % '')
      rowcount += 1


# ---------------------------------------------------------------------------
def cmdstatus(showall: bool = False):
  """ Print current parameters """
  # global Prompt, conversation
  global UpdateCheck, token_limit, used_tokens
  printinfo(f'DéjàVu GPT Terminal vs {Version} |  Enter ! for help.')
  def pp(pref: str, suff: str):
    printinfo(f'%16s: {suff}' % pref)
  pp(   'Agent',            (SYSTEM_NAME + (' '+ConvFile.replace(HOME, '~')) if showall else SYSTEM_NAME))
  pp(   'User Name',        USER_NAME)
  pp(   'Assistant Name',   AI_NAME)
  pp(   'AI Engine',        engine)
  pp(   'AutoSave',         'On' if AutoSave else 'Off')
  if showall:
    token_limit = 4000 if (engine in [ 'text-davinci-003', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4']) else 2000
    used_tokens = num_tokens_from_string(Prompt + ''.join(conversation))
    pp( 'Token Limit',      token_limit)
    pp( 'Response Tokens',  response_tokens)
    pp( 'Tokens Used',      int(used_tokens))
    pp( 'Tokens Left',      int(token_limit - used_tokens))
    pp( 'Temperature',      temperature)
    pp( 'Top_p',            top_p)
    pp( 'Frequency',        freq_pen)
    pp( 'Presence',         pres_pen)
    pp( 'Stop',             str(stop) if len(stop) else 'None')
    pp( 'Command Echo',     'On' if cmdEcho else 'Off')
    pp( 'Shell',            os.path.basename(SHELL))
    pp( 'Editor',           os.path.basename(EDITOR))
    pp( 'Browser',          os.path.basename(BROWSER))
    pp( 'Command History',  historyFile.replace(HOME, '~'))
  if not UpdateCheck:
    try:
      rstat = requests.get(REPOSITORY_VERSION, timeout=2, headers={'User-Agent': 'Wget/1.21.2', 'Accept': '*/*', 'Accept-Encoding': 'identity', 'Connection': 'Keep-Alive'})
      data = str(rstat.text).strip()
      if Version < data:
        pp('Update', f'Version {data} of DéjàVu is now available.')
        pp('', f"Run '{ScriptName} -q --update' to update.")
      UpdateCheck = True
    except:
      pass


def role_tag_replace(text: str) -> str:
  """
  Replace role tag with role name. 
  global SYSTEM_NAME, AI_NAME, USER_NAME
  """
  return text.replace('<<AI_NAME>>', AI_NAME).replace('<<USER_NAME>>', USER_NAME).replace('<<SYSTEM_NAME>>', SYSTEM_NAME)

def role_name_replace(text: str) -> str:
  """
  Replace role name with role tag. 
  global SYSTEM_NAME, AI_NAME, USER_NAME
  """
  return text.replace(AI_NAME, '<<AI_NAME>>').replace(USER_NAME, '<<USER_NAME>>').replace(SYSTEM_NAME, '<<SYSTEM_NAME>>')


# ---------------------------------------------------------------------------
def PromptReplace(conversationText: str='', AIs: str='<<AI_NAME>>', AIr: str='', USs: str='<<USER_NAME>>', USr: str='', SYs: str='<<SYSTEM_NAME>>', SYr: str='') -> str:
  """
  Replace <<>> markers in prompt and conversation
  with the appropriate text
  """
  # global Prompt, AI_NAME, USER_NAME
  if not SYr: SYr = SYSTEM_NAME
  if not AIr: AIr = AI_NAME

  if not USr: USr = USER_NAME
  newp = Prompt + ('\n' if Prompt[-1] != '\n' else '') \
          + conversationText
  newp = newp.replace(AIs, AIr, -1).replace(USs, USr, -1).replace(SYs, SYr, -1)
  return newp.strip() + '\n'

# ----------------------------------------------------------------------------
def autoSave(dv_filename: str, auto=True):
  """ Autosave current chat """
  # global conversation, Instructions,
  if cmdEcho: printinfo(('Auto' if auto else '') + 'Saving ' + dv_filename, end='')
  autofound = False
  autosave_file = tempname('autosave', '.dv')
  writefile(autosave_file, '', 'w')
  for instr in Instructions:
    # comments or blank lines, just insert now
    if len(instr) == 0 or instr[0] == '#':
      writefile(autosave_file, f'{instr}\n', 'a')
      continue

    # !cmd forgiveness, but standardize on /cmd.
    if instr[0] == '!': instr = '/' + instr[1:]

    instr_tok = tokenize(instr)
    if len(instr_tok) == 0:
      writefile(autosave_file, '\n', 'a')
      continue
    instr_tok[0] = instr_tok[0].lower()
    if instr_tok[0].startswith('/temp'):
      writefile(autosave_file, f'/temperature {temperature}\n', 'a')
      continue
    if instr_tok[0].startswith('/engi'):
      writefile(autosave_file, f'/engine {engine}\n', 'a')
      continue
    if instr_tok[0].startswith('/ai_n'):
      writefile(autosave_file, f'/ai_name {AI_NAME}\n', 'a')
      continue
    if instr_tok[0].startswith('/user'):
      writefile(autosave_file, f'/user_name {USER_NAME}\n', 'a')
      continue
    if instr_tok[0].startswith('/top_p'):
      writefile(autosave_file, f'/top_p {top_p}\n', 'a')
      continue
    if instr_tok[0].startswith('/freq'):
      writefile(autosave_file, f'/freq_pen {freq_pen}\n', 'a')
      continue
    if instr_tok[0].startswith('/pres'):
      writefile(autosave_file, f'/pres_pen {pres_pen}\n', 'a')
      continue
    if instr_tok[0].startswith('/auto'):
      writefile(autosave_file, f'/autosave {("On" if AutoSave else "Off")}\n', 'a')
      autofound = True
      continue
#    if instr_tok[0].startswith('/echo'):
#      writefile(autosave_file, f'/echo {("On" if cmdEcho else "Off")}\n', 'a')
#      continue
    if instr_tok[0].startswith('/stop'):
      writefile(autosave_file, f'/stop {str(stop)}\n', 'a')
      continue
    if instr_tok[0].startswith('/prom') or instr_tok[0].startswith('!prom'):
      instr = '/prompt """\n' \
              + ' '.join(instr_tok[1:]).replace('\\n', '\n') \
              + '\n"""\n'
      writefile(autosave_file, instr, 'a')
      continue
    # ignore /conversation for now in Instructions Section
    if instr_tok[0].startswith('/conv') or instr_tok[0].startswith('!conv'):
      continue
    # an explcit instruction to gpt
    if instr_tok[0].startswith('/inst'):
      instr = '/instruction """\n' \
              + ' '.join(instr_tok[1:]).replace('\\n', '\n') \
              + '\n"""\n'
      writefile(autosave_file, instr, 'a')
      continue
    # a generic instruction or other /command
    writefile(autosave_file, instr + '\n', 'a')

  if not autofound:
    writefile(autosave_file, f'/autosave {("On" if AutoSave else "Off")}\n', 'a')

  # now write out the complete new conversation
  for instr in conversation:
    writefile(autosave_file, '/conversation ' + instr.replace('\n', '\\n') + '\n', 'a')
  
  try:
    # rename files
    if os.path.exists(dv_filename):
      os.replace(dv_filename, dv_filename + '~')
    # replace current script with new script    
    os.rename(autosave_file, dv_filename)
  except Exception as saveerr:
    printerr(f'Error renaming {autosave_file} to {dv_filename}: {str(saveerr)}')

# ----------------------------------------------------------------------------
def orderly_exit():
  """ Exit script nicely """
  # global AutoSave, Verbose
  if AutoSave: autoSave(ConvFile)
  if Verbose: 
    print('\r\x1b[2K' if UseColor() else '', end='')
    printinfo(f'Exiting {ScriptName} {ConvFile}')
    print(Style.RESET_ALL if UseColor() else '', end='')
  sys.exit(0)


# ----------------------------------------------------------------------------
if __name__ == '__main__':
  conversation = []
  while True:
    if len(cmdTypeAhead) == 0 and cmdExit: sys.exit(0)
    getScreenColumns()
    used_tokens = num_tokens_from_string(Prompt + ''.join(conversation))

    if len(cmdTypeAhead) > 0:
      userInput = cmdTypeAhead.pop(0)
      if userInput[0].rstrip() == '#' or len(userInput.strip()) == 0: continue
      if cmdEcho and (userInput[0:5] != '/echo' and userInput[0:5] != '!echo'):
        printstd(USER_NAME + ':', color=Fore.YELLOW + Style.BRIGHT)
        print(userInput)
    else:
      cmdEcho = True
      printstd(f'{len(conversation)+1:d}. {USER_NAME}:', color=Fore.YELLOW + Style.BRIGHT)
      try:
        userInput = input().lstrip()
      except KeyboardInterrupt:
        if input_key('Exit Dejavu?') == 'y': orderly_exit()
        continue

    if len(userInput.strip()) == 0 or userInput[0].rstrip() == '#': continue

    # Process /command
    if userInput[0] == '!' or userInput[0] == '/':
      command = userInput[1:].strip()
      tok = tokenize(command)

      # help
      if len(tok) == 0:
        cmd_help()
        continue

      tok[0] = tok[0].lower()

      # /help|/?|/!|//
      if tok[0] in ['help', '?', '/']:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
          subprocess.run(['man', 'dv'], check=False)
        except KeyboardInterrupt:
          print('^C', file=sys.stderr)
          pass
        except Exception as e:
          printerr('Error running `man`.', e)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        continue

      # /exit
      if tok[0] in ['exit', 'quit']:
        orderly_exit()
        continue

      # /chdir|/cd
      if tok[0] in ['cd', 'chdir', 'chdi']:
        if len(tok) > 1:
          if tok[1] == 'list':
            while True:
              directories = get_directories()
              directories.append('..')
              tmp = selectList(directories, 'ChDir ')
              if not tmp: break
              try: os.chdir(tmp)
              except Exception as e:
                printerr(f'Could not change directory into "{tmp}"', e)
          else:
            try: os.chdir(' '.join(tok[1:]))
            except Exception as e: 
              printerr(f'Could not change directory into "{" ".join(tok[1:])}"', e)
        if cmdEcho: 
          printinfo('Current directory is ' + os.getcwd())
        continue

      # /syntax lang func
      if tok[0] in ['synt', 'syntax']:
        if len(tok) < 3:
          printerr('Requires options')
          printinfo('/syntax language cmd|function', 'eg, /syntax php strpos')
          continue
        cmdTypeAhead = [f'In {tok[1]}, list complete synopsis, syntax, parameters, options, keywords, and usage for "{tok[2]}", giving coded examples: ']
        continue

      # /echo [on|off]
      if tok[0] == 'echo':
        if len(tok) > 1:
          tok[1] = tok[1].lower()
          if tok[1] in ['1', 'on', 'true']:
            cmdEcho = True
          elif tok[1] in ['0', 'off', 'false']:
            cmdEcho = False
          else:
            printerr('Invalid argument')
        else: printinfo('Command Echo is ' + str('On' if cmdEcho else 'Off'))
        continue

      # /rawoutput [on|off*]
      if tok[0] in ['rawo', 'rawoutput']:
        if len(tok) > 1:
          tok[1] = tok[1].lower()
          if tok[1] in ['1', 'on', 'true']:
            RawOutput = True
          elif tok[1] in ['0', 'off', 'false']:
            RawOutput = False
          else:
            printerr('Invalid argument')
        else: print('# Raw Output is ' + str('On' if RawOutput else 'Off'), end='\n')
        continue

      # /autosave [on|off]
      if tok[0] in ['auto', 'autosave']:
        if len(tok) > 1:
          tok[1] = tok[1].lower()
          if tok[1] in ['1', 'on', 'true']:
            AutoSave = True
          elif tok[1] in ['0', 'off', 'false']:
            AutoSave = False
          else:
            printerr('Invalid argument')
        else: printinfo('AutoSave is ' + str('On' if AutoSave else 'Off'))
        continue

      # /scripts /agents /files
      if tok[0] in ['scri', 'scripts', 'file', 'files', 'agen', 'agents']:
        script = selectFile(['.', dvHome], '*.dv', 'Select Script')
        if not script: continue
        try:
          subprocess.run(f'{EDITOR} "{script}"', shell=True, executable=SHELL, check=False)
          if input_key(f'Run {script}?') != 'y': continue
          cmdTypeAhead = [f'/run {script}']
        except KeyboardInterrupt:
          print('^C', file=sys.stderr)
          continue
        except Exception as e:
          printerr('ERROR in editor.', str(e)); continue
        continue

      # /edit
      if tok[0] == 'edit':
        modify_datestamp = os.path.getmtime(ConvFile)
        try:
          subprocess.run(f'{EDITOR} {ConvFile}', shell=True, executable=SHELL, check=False)
        except Exception as e:
          printerr('Edit error ' + str(e)); continue
        if modify_datestamp != os.path.getmtime(ConvFile):
          if input_key(f'Re-Load {ConvFile}?') == 'y':
            cmdTypeAhead = ['/run ' + ConvFile]
        continue

      # /exec||!! [program [args]]
      if tok[0] in ['exec', '!']:
        try:
          if len(tok) == 1:
            execstr = SHELL
            os.environ['PS1'] = "\rDejaVu:" + os.environ['PS1']
          else:
            execstr = ' '.join(tok[1:])
          subprocess.run(execstr, shell=True, executable=SHELL, check=False)
        except KeyboardInterrupt:
          print('^C', file=sys.stderr)
          continue
        except Exception as e:
          printerr('Exec error ' + str(e))
        continue

      # /status [short|long]
      if tok[0] in ['stat', 'status']:
        longstatus = True
        if len(tok) > 1:
          if tok[1] == 'short': longstatus = False
        cmdstatus(longstatus)
        continue

      # /username [name]
      if tok[0] in ['user', 'user_name']:
        if len(tok) > 1:
          tmp = re.sub(r'[^a-zA-Z0-9_-]', '', '-'.join(tok[1:])).strip().upper()
          if len(tmp) < 4 or len(tmp) > 16:
            printerr('Invalid length in user_name "' + tmp + '". Min 4, Max 16.')
            continue
          USER_NAME = tmp
        if cmdEcho: printinfo('USER_NAME is now ' + USER_NAME)
        continue
      # /ai_name [name]
      if tok[0] in ['ai_n', 'ai_name']:
        if len(tok) > 1:
          tmp = re.sub(r'[^a-zA-Z0-9_-]', '', '-'.join(tok[1:])).strip().upper()
          if len(tmp) < 4 or len(tmp) > 16:
            printerr('Invalid length in ai_name "' + tmp + '". Min 4, Max 16.')
            continue
          AI_NAME = tmp
        if cmdEcho: printinfo('AI_NAME is now ' + AI_NAME)
        continue

      # /engine [list|engine|update]
      if tok[0] in ['engi', 'engine', 'engines']:
        if tok[0] == 'engines' and len(tok) == 1: tok.append('list')
        if len(tok) > 1:
          if tok[1] == 'list' or tok[1] == 'select':
            with open(f'{ScriptDir}/engines.json') as f:
                data = json.load(f)
            gptengines = []
            for item in data['data']:
                gptengines.append(item['id'])
            gptengines.sort()
            tmp = selectList(gptengines, 'Select GPT Engine')
            gptengines = None
            if len(tmp) > 0:
              engine = tmp
          elif tok[1] == 'update':
            printinfo('Updating GPT engine list...')
            tmpfile = tempname('engines.list')
            enginesfile = f'{ScriptDir}/engines.json'
            try:
              subprocess.run(f'openai api engines.list',
                  stdout=open(tmpfile, 'w'),
                  shell=True, executable=SHELL, check=False)
              os.rename(enginesfile, enginesfile+'~')
              os.rename(tmpfile, enginesfile)
              printinfo('GPT engine list updated.')
            except:
              printerr('GPT engine list not updated.')
              pass
            continue
          else:
            engine = tok[1]
        token_limit = 4000 if (engine in [ 'text-davinci-003', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4']) else 2000
        if response_tokens > (token_limit-100):
          response_tokens = int(token_limit/2)
        if cmdEcho:
          printinfo(f'Engine is now {engine}')
          printinfo(f'Max Allowed Tokens: {token_limit}, Max Response Tokens: {response_tokens}')
        continue
      # /temperature [1.0]
      if tok[0] in ['temp', 'temperature']:
        if len(tok) > 1 and is_num(tok[1]):
          temperature = max(0.0, min(1.0, float(tok[1])))
        if cmdEcho: printinfo('Temperature is now ' + str(temperature))
        continue
      # /top_p [1.0]
      if tok[0] in ['top_', 'top_p']:
        if len(tok) > 1 and is_num(tok[1]):
          top_p = max(0.0, min(1.0, float(tok[1])))
        if cmdEcho: printinfo('Top_p is now ' + str(top_p))
        continue
      # /tokens [number]
      if tok[0] in ['toke', 'tokens']:
        if len(tok) > 1 and is_num(tok[1]):
          if int(tok[1]) > 0:
            response_tokens = max(16, min(token_limit - used_tokens - 16, int(tok[1])))
          else:
            response_tokens = -1
        if cmdEcho: 
          printinfo('Max Tokens is now ' + str(response_tokens))
        continue
      # /freq_pen [0.0]
      if tok[0] in ['freq', 'freq_pen']:
        if len(tok) > 1 and is_num(tok[1]):
          freq_pen = max(-2, min(2, float(tok[1])))
        if cmdEcho: printinfo('Frequency Penalty is now ' + str(freq_pen))
        continue
      # /pres_pen [0.0]
      if tok[0] in ['pres', 'pres_pen']:
        if len(tok) > 1 and is_num(tok[1]):
          pres_pen = max(-2, min(2, float(tok[1])))
        if cmdEcho: printinfo('Presence Penalty is now ' + str(pres_pen))
        continue

      # /spacetime [location]
      if tok[0] in ['spac', 'spacetime']:
        temp = f"Today's date is {datetime.datetime.today().strftime('%Y-%m-%d')}, and current {USER_NAME} local time is {datetime.datetime.now((time.tzname[0] if time.timezone >= 0 else time.tzname[1])).strftime('%H:%M %Z')}."
        if len(tok) > 1:
          temp += f" Current {USER_NAME} location is {tok[1:]}."
        Prompt = Prompt + '\n\n## SPACETIME\n\n' + temp + '\n'
        # Add the response to the conversation list
        conversation.append(f'<<AI_NAME>>: I acknowledge: {temp}')
        continue

      # /system [/append*|/insert|/replace prompt]
      if tok[0] in ['syst', 'system']:
        tok[0] = '/prompt'
      # /prompt [/append*|/insert|/replace prompt]
      if tok[0] in ['prom', 'prompt']:
        if len(tok) > 1:
          if tok[1].startswith('/appe') or tok[1].startswith('/add'):
            Prompt = Prompt + '\n' \
                      + ' '.join(tok[2:]).replace('\\n', '\n').strip()
          elif tok[1].startswith('/inse') or tok[1].startswith('/insert'):
            Prompt = ' '.join(tok[2:]).replace('\\n', '\n').strip() \
                      + '\n' + Prompt
          elif tok[1].startswith('/repl') or tok[1].startswith('/replace'):
            Prompt = ' '.join(tok[2:]).replace('\\n', '\n').strip() 
          else:
            Prompt = Prompt + '\n' \
                      + ' '.join(tok[1:]).replace('\\n', '\n').strip()

        elif cmdEcho: printinfo(Prompt, prefix='')
        continue

      # /conversation [conversation]
      if tok[0] in ['conv', 'conversation']:
        if len(tok) > 1:
          conversation.append(' '.join(tok[1:]).replace('\\n', '\n'))
        else:
          for tmp in conversation:
            printinfo(tmp, prefix='')
        continue

      # /list [0-0]
      if tok[0] == 'list':
        short = False
        if len(tok) > 1:
          if   tok[1][0:4] == 'shor': short = True; tok.pop(1)
          elif tok[1][0:4] == 'long': short = False; tok.pop(1)
        if len(tok) < 2: tok.append('all')
        rnge = int_list(tok[1:], 1, len(conversation), False)
        if not rnge: continue
        for rindex in rnge:
          text = conversation[rindex - 1].replace('<<USER_NAME>>', USER_NAME).replace('<<AI_NAME>>', AI_NAME)
          if text[0:len(USER_NAME) + 2] == USER_NAME + ': ':
            printstd(f'{rindex:d}. {USER_NAME}:', color=Fore.YELLOW, style=Style.DIM)
            text = text[len(USER_NAME) + 2:]
          elif text[0:len(AI_NAME) + 2] == AI_NAME + ': ':
            printstd(f'{rindex:d}. {AI_NAME}:', color=Fore.GREEN, style=Style.DIM)
            text = text[len(AI_NAME) + 2:] + '\n\n'
          for _line in text.splitlines():
            if short:
              printinfo(_line[0:ScreenColumns - 3] + '...', prefix='')
              break
            printinfo(textwrap.fill(_line, width=ScreenColumns), prefix='')
        continue

      # /clear [conversation*|prompt|all]
      if tok[0] in ['clea', 'clear']:
        if len(tok) == 1 or (tok[1] in ['conv', 'conversation']):
          conversation = []
          if cmdEcho: printinfo('All conversation has been cleared.')
        elif tok[1] in ['prom', 'prompt']:
          prompt = ''
          Prompt = ''
          if cmdEcho: printinfo('Prompt has been cleared.')
        elif tok[1] == 'all':
          prompt = ''
          Prompt = ''
          conversation = []
          if cmdEcho: printinfo('Prompt and conversation have been cleared.')
        continue

      # /delete 0-0
      if tok[0] in ['dele', 'delete']:
        if len(tok) < 2:
          printerr('Range not specified.')
          continue
        rnge = int_list(tok[1:], 1, len(conversation), True)
        if not rnge: continue
        i = int(0)
        for rdel in rnge:
          del conversation[rdel - 1]
          i += 1
        if cmdEcho: printinfo(f'{i:d} entries deleted')
        continue

      # /tldr [range]
      if tok[0] in ['tldr', 'tl;dr']:
        if len(tok) < 2: tok.append(str(len(conversation)))
        rnge = int_list(tok[1:], 1, len(conversation), False)
        if not rnge: continue
        i = int(0)
        for rtldr in rnge:
          text = conversation[rtldr - 1]
          text = text.replace('<<AI_NAME>>', AI_NAME).replace('<<USER_NAME>>', USER_NAME).replace('<<SYSTEM_NAME>>', SYSTEM_NAME)
          try:
            tldr_response = gpt35_completion(
                text + '\n\nTL;DR: ',
                [], 
                engine='text-davinci-003', 
                temperature=temperature, 
                top_p=top_p, 
                tokens=response_tokens, 
                freq_pen=freq_pen, 
                pres_pen=pres_pen, 
                stop=stop)
            if len(tldr_response) == 0: continue
          except:
            printerr('GPT experienced an error. Possibly overloaded.')
            continue
          printstd(AI_NAME  + ': ', color=Fore.GREEN)
          tldr_response = 'TLDR'+ ('' if len(rnge) == 1 else f'[{str(i+1)}]') + ': ' + tldr_response
          printinfo(tldr_response, prefix='')
          conversation.append(f'<<AI_NAME>>: {tldr_response}')
          i += 1
        continue

      # /save [script]
      if tok[0] == 'save':
        if len(tok) < 2: filename = ConvFile
        else: filename = tok[1].replace('"', '').replace('"', '')
        filename = find_dv_file(filename, mustexist=False)
        if os.path.exists(filename):
          if input_key(f'\n{filename} exists. Overwrite?') != 'y':
            continue
        autoSave(filename, False)
        if cmdEcho: printinfo('Session saved to ' + filename)
        continue

      # /vars | /locals
      if tok[0] in ['vars', 'locals']:
        varsearch = tok[1] if len(tok) > 1 else ''
        locals_list = [(key, value) for key, value in locals().items() if not callable(value)]
        for key, value in locals_list:
          value = str(value).replace('\n', '\\n')
          if value[0:7] == '<module': continue
          if varsearch and key[0:len(varsearch)] != varsearch: continue
          print(f"{key}={value}")
        print('sys.path=', sys.path)
        print()
        print('Environment:')
        for env in ['USER', 'HOME', 'SHELL', 'EDITOR', 'BROWSER', 'TMPDIR', 'PATH', 'OPENAI_API_KEY', 'OPENAI_ORGANIZATION_ID']:
          print(f'{env}={str(os.getenv(env))}')
        continue

      # /history [range]
      if tok[0] in ['hist', 'history']:
        if len(tok) > 1:
          if is_num(tok[1][0]):
            rnge = int_list(tok[1:], 1, readline.get_current_history_length(), False)
            if not rnge: continue
            for rhis in rnge:
              cmdTypeAhead.append(readline.get_history_item(int(rhis)))
          else:
            printerr(f'Unknown parameter {tok[1]}.')
          continue
        history_length = readline.get_current_history_length()
        prev_item = ''
        for i in range(history_length, 0, -1):
          item = readline.get_history_item(i)
          if len(item) <= 2:
            readline.remove_history_item(i - 1)
            continue
          if item == prev_item:
            readline.remove_history_item(i - 1)
            continue
          prev_item = item
        history_length = readline.get_current_history_length()
        for i in range(1, history_length + 1):
          print(i, readline.get_history_item(i))
        continue

      # /awesome [update]
      if tok[0] in ['awes', 'awesome']:
        curdir = os.getcwd()
        os.chdir(ScriptDir)
        awe_prompt = select_awesome_prompt(tok[0:])
        os.chdir(curdir)
        if len(awe_prompt) == 0: continue
        print(awe_prompt)
        userInput = awe_prompt

      # /run [script]
      elif tok[0] == 'run':
        if len(tok) < 2:
          script = selectFile(['.', dvHome], '*.dv', 'Select Script to Run')
          if not script: continue
        else:
          script = find_dv_file(' '.join(tok[1:]), mustexist=True)
          if len(script) == 0:
            printerr(f'dvScript {script} does not exist.')
            continue
        try:
          if AutoSave: autoSave(ConvFile)
          readline.write_history_file(historyFile)
          read_dvfile(script)
          ConvFile = script
          historyFile = initHistory(ConvFile)
        except Exception:
          printerr(f'Run failed for "{script}". Reloading "{ConvFile}".')
          read_dvfile(ConvFile)
          historyFile = initHistory(ConvFile)
        continue

      # /instruction [instruction]
      elif tok[0] in ['inst', 'instruction']:
        if len(tok) > 1:
          userInput = ' '.join(tok[1:]).replace('\\n', '\n').strip()
        else:
          for tmp in Instructions:
            if tmp[0:8] == '/prompt ' or tmp[0:8] == '!prompt ':
              tmp = '/prompt """\n' + \
                    tmp[8:].replace('\\n', '\n') + \
                    '"""'
            elif tmp[0:13] == '/instruction ' or tmp[0:13] == '!instruction ':
              tmp = '/instruction """\n' + \
                    tmp[13:].replace('\\n', '\n') + '\n' + \
                    '"""'
            printinfo(tmp, prefix='')
          continue
        # proceed to gpt

      # /import [filename]
      elif tok[0] in ['impo', 'import']:
        if len(tok) < 2:
          tmpfile = tempname('command', '.dv')
          writefile(tmpfile, '')
          while True:
            try:
              subprocess.run(f'{EDITOR} {tmpfile}', shell=True, executable=SHELL, check=False)
            except Exception as e:
              printerr('Import file error.', str(e))
              userInput = ''
              break
            userInput = readfile(tmpfile)
            if len(userInput) == 0: break
            ynr = input_key(f'Execute instructions in {tmpfile}? Or re-edit?', ['y', 'n', 'r'], ['n'])
            if ynr == 'y': break
            if ynr == 'r': continue
            userInput = ''
            break
          try:
            os.remove(tmpfile)
            os.remove(tmpfile + '~')
          except FileNotFoundError: pass
          if not userInput: continue
          print('\n' + userInput.strip())
        # proceed to gpt
        else:
          filename = tok[1]
          if not os.path.exists(filename):
            printerr(filename + ' does not exist.')
            continue
          if cmdEcho: printinfo('Importing from text file ' + filename)
          userInput = readfile(filename).strip() + '\n'
        # proceed to gpt

      # /summarise [conv|prompt|*all]
      elif tok[0] in ['summ', 'summarize', 'summarise']:
        if len(tok) < 2: what = 'conversation'
        else: what = tok[1]
        if what[0:4] == 'prom' or what == 'prompt':
          userInput = PromptReplace() \
              + f'\n\n{USER_NAME}: Write a detailed summary of all the above: '
        elif what[0:4] == 'conv' or what == 'conversation':
          userInput = text_block \
              + f'\n\n{USER_NAME}: Write a detailed summary of all the above: '
        elif what == 'all':
          userInput = PromptReplace(text_block) \
              + f'\n\n{USER_NAME}: Write a detailed summary of all the above: '
        else:
          printerr('Invalid option. Valid options for /summarize are prompt|conv|all.')
          continue

        # proceed to gpt

      # Invalid command
      else:
        printerr('Invalid command: /' + command)
        if Verbose: printinfo("/ or /help for command help.")
        continue

    # Prepend username to user input
    conversation.append(f'<<USER_NAME>>: {userInput}')
    # Aggregate the entire conversation
    text_block = ('\n'.join(conversation)).strip('\n') \
                 + '\n' + AI_NAME + ': '

    # Send the entire conversation to GPT-3
    try:
      response = gpt35_completion(
          (PromptReplace(text_block) if engine not in [ 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4' ] else PromptReplace()), 
          conversation,
          engine=engine, temperature=temperature, top_p=top_p, 
          tokens=response_tokens, freq_pen=freq_pen, pres_pen=pres_pen, stop=stop)
      if len(response) == 0:
        conversation.pop()
        continue
    except Exception as e:
      printerr('GPT experienced an error. Possibly overloaded.', e)
      conversation.pop()
      continue
    # Add the response to the conversation list
    conversation.append(f'<<AI_NAME>>: {response}')
    if RawOutput:
      print(response, end='\n')
    else:
      # Print the response from GPT-3
      printstd(f'{len(conversation):d}. {AI_NAME}:', color=Fore.GREEN)
      for _line in response.splitlines():
        print(textwrap.fill(_line, width=ScreenColumns))

    # eg, devaju -x -c 'command'
    if len(cmdTypeAhead) == 0 and cmdExit: sys.exit(0)

# end
