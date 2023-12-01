#!/usr/bin/env python3

import io
import json
import logging
import os
import readline
import shlex
import sys
import tempfile

import pydantic
import openai

import internetarchive as ia

from langchain.llms import OpenAI
from llama_index import GPTSimpleVectorIndex, GPTListIndex, Document, LLMPredictor, SimpleDirectoryReader, ServiceContext
from requests.exceptions import HTTPError
from rich.console import Console
from urllib.parse import urlparse

if not __package__:
    sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from iacopilot import __NAME, __VERSION


logger = logging.getLogger("llama_index")
logger.setLevel(logging.WARNING)


class TabCompleter():
  def __init__(self, msg=""):
    self.tree = {}
    for tokens in msg.splitlines():
      for sep in ["  ", "<", "[", "/", "!"]:
        tokens, _, _ = tokens.partition(sep)
      self.add(*tokens.split())

  def add(self, *args):
    root = self.tree
    for arg in args:
      if arg not in root:
        root[arg] = {}
      root = root[arg]

  def remove(self, *args, children=False):
    root = self.tree
    for arg in args[:-1]:
      if arg not in root:
        return
      root = root[arg]
    if args[-1] in root:
      if children:
        root[args[-1]] = {}
      else:
        del root[args[-1]]

  def completer(self, text, state):
    line = readline.get_line_buffer()
    tokens = line.split()
    if line[-1:] != " ":
      tokens = tokens[:-1]
    root = self.tree
    try:
      for t in tokens:
        root = root[t]
    except KeyError as e:
      root = {}
    return [t + " " for t in filter(lambda x: x.startswith(text), root.keys())][state]


class IaCopilot():
  HELP = """
help/h/?                    Print this help message
quit/exit/q                 Exit the REPL prompt
ls                          List all the loaded contexts
load <URL>                  Detect source and load the data as a context
load ia <ITEM>              Load an IA item as a context
load tv <CHANNEL> [<DATE>]  Load transcript of a TV channel as a context
load wbm <URL> [<DATE>]     Load a Wayback Machine capture as a context
load wiki <TITLE> [<LANG>]  Load a Wiki page as a context
load file <PATH>            Load a loal file or directory as a context
cd [<ID>]                   Change a loaded context to query
rm [<ID>]                   Remove current or specified context
reset                       Remove all contexts and reset statistics
config openai [<KEY>]       Get or set configuration options
<PROMPT>                    Ask the copilot questions about the context
! <CMD>                     Run a system command
""".strip()

  QUITS = {"q", "quit", "exit"}
  HELPS = {"?", "h", "help"}
  TVCHANS = ["ESPRESO", "RUSSIA1", "RUSSIA24", "1TV", "NTV", "BELARUSTV", "IRINN"]

  def annotate(self, msg=""):
    BRACKETS = {"<", "["}
    lines = []
    for line in msg.splitlines():
      cmd, sep, des = line.partition("  ")
      tb = "/" if "/" in cmd else " "
      tokens = []
      for t in cmd.split(tb):
        if t[:1] in BRACKETS:
          tokens.append(t.replace("<", "[magenta]<").replace(">", ">[/]"))
        else:
          tokens.append(f"[cyan]{t}[/]")
      lines.append(f" {tb.join(tokens)}{sep}{des}")
    return "\n".join(lines)

  def __init__(self, name="IACopilot", version="0.0.0", ps="[bold red]{name}[/] [magenta]{queries}:{tokens}[/] [cyan]{uri}[/]> "):
    self.store = {}
    self.name = name
    self.version = version
    self.ps = ps
    self.queries = 0
    self.tokens = 0
    self.context = ""
    self.id = ""
    self.buf = io.StringIO()
    self.bufcon = Console(file=self.buf, force_terminal=True, highlight=False, soft_wrap=True)
    self.console = Console(highlight=False, soft_wrap=True)
    self.pretty_help = self.annotate(msg=self.HELP)
    self.tc = TabCompleter(msg=self.HELP)
    for ch in self.TVCHANS:
      self.tc.add("load", "tv", ch)
    self.llmp = LLMPredictor(llm=OpenAI(max_tokens=1024, model_name="text-davinci-003"))
    self.sctx = ServiceContext.from_defaults(llm_predictor=self.llmp)

  @property
  def uri(self):
    return "?" if not self.context else f"{self.context}:{self.id}"

  @property
  def prompt(self):
    params = {
      "name": self.name,
      "version": self.version,
      "queries": self.queries,
      "tokens": self.tokens if self.tokens < 1000 else f"{round(self.tokens / 1000)}k",
      "context": self.context,
      "id": self.id,
      "uri": self.uri
    }
    return self.colorize(self.ps.format(**params))

  def colorize(self, msg):
    self.buf.seek(0)
    self.buf.truncate()
    self.bufcon.print(msg, end="")
    return self.buf.getvalue()

  def welcome(self):
    self.console.print("Enter [cyan]quit[/]   to quit/exit this REPL prompt")
    self.console.print("Enter [cyan]help[/]   to print the help message")
    self.console.print("Press [cyan]<TAB>[/]  to see available commands")

  def print_help(self):
    self.console.print(self.pretty_help)

  def print_version(self):
    self.console.print(f"[cyan]{self.name}[/] [magenta]{self.version}[/]")

  def isurl(self, url):
    try:
      u = urlparse(url)
      return bool(u.netloc) and u.scheme in {"http", "https"}
    except:
      return False

  def parse(self, p):
    if p in self.QUITS:
      raise SystemExit("User quits the REPL")
    if p in self.HELPS:
      return self.print_help()
    if p[:1] == "!":
      return os.system(p[1:].strip())
    tokens = shlex.split(p)
    length = len(tokens)
    if tokens[0] == "ls" and length == 1:
      return self.list()
    if tokens[0] == "reset" and length == 1:
      return self.reset()
    if tokens[0] == "load" and length == 2 and self.isurl(tokens[1]):
      return self.load_url(tokens[1])
    if tokens[0] == "load" and length == 2 and tokens[1] == "ia":
      self.console.print("[red]A valid IA item ID parameter is required![/]")
      self.console.print("Usage: [cyan]load ia[/] [magenta]<ITEM>[/]")
      return
    if tokens[0] == "load" and length == 3 and tokens[1] == "ia":
      return self.load_ia(tokens[2])
    if tokens[0] == "load" and length == 2 and tokens[1] == "tv":
      self.console.print("[red]A valid TV channel ID parameter is required (and optional DATE)![/]")
      self.console.print("Usage: [cyan]load tv[/] [magenta]<CHANNEL> [<DATE>][/]")
      return
    if tokens[0] == "load" and 2 < length < 5 and tokens[1] == "tv":
      return self.load_tv(*tokens[2:4])

    if tokens[0] == "load" and length == 2 and tokens[1] == "wbm":
      self.console.print("[red]A valid URL parameter is required (and optional DATE)![/]")
      self.console.print("Usage: [cyan]load wbm[/] [magenta]<URLL> [<DATE>][/]")
      return
    if tokens[0] == "load" and 2 < length < 5 and tokens[1] == "wbm" and self.isurl(tokens[2]):
      return self.load_wbm(*tokens[2:4])

    if tokens[0] == "load" and length == 2 and tokens[1] == "wiki":
      self.console.print("[red]A valid wiki title parameter is required (and optional language)![/]")
      self.console.print("Usage: [cyan]load wiki[/] [magenta]<TITLE> [<LANG>][/]")
      return
    if tokens[0] == "load" and 2 < length < 5 and tokens[1] == "wiki":
      return self.load_wiki(*tokens[2:4])
    if tokens[0] == "load" and length == 2 and tokens[1] == "file":
      self.console.print("[red]A valid filre or folder path parameter is required![/]")
      self.console.print("Usage: [cyan]load file[/] [magenta]<PATH>[/]")
      return
    if tokens[0] == "load" and length == 3 and tokens[1] == "file":
      return self.load_file(tokens[2])
    if tokens[0] == "config" and length == 2 and tokens[1] == "openai":
      return self.get_openai_key()
    if tokens[0] == "config" and length == 3 and tokens[1] == "openai":
      return self.set_openai_key(tokens[2])
    if tokens[0] == "cd" and length < 3:
      return self.change_context(*tokens[1:2])
    if tokens[0] == "rm" and length < 3:
      return self.rm_context(*tokens[1:2])
    self.ask_gpt(p)

  def list(self):
    if not self.store:
      self.console.print("[red]No context is loaded yet![/]")
      self.console.print("Usage: [cyan]load ia[/] [magenta]<ITEM>[/]")
      return
    for ctx in self.store:
      self.console.print(ctx)

  def reset(self):
    self.clear_context()
    self.store = {}
    self.queries = 0
    self.tokens = 0
    self.tc.remove("cd", children=True)
    self.tc.remove("rm", children=True)

  def clear_context(self):
    self.id = ""
    self.context = ""

  def change_context(self, id=""):
    if not id:
      self.clear_context()
      return
    if not id in self.store:
      return
    self.id = id
    self.context = self.store[id]["context"]

  def add_context(self, id, ctx):
    self.store[id] = ctx
    self.change_context(id)
    self.tc.add("cd", id)
    self.tc.add("rm", id)

  def rm_context(self, id=""):
    if not id and self.id:
      id = self.id
    if not id:
      return
    if id == self.id:
      self.clear_context()
    if id in self.store:
      del self.store[id]
      self.tc.remove("cd", id)
      self.tc.remove("rm", id)

  def load_url(self, url):
    self.console.print("[red]Not implemented yet![/]")

  def load_ia(self, id):
    if id in self.store:
      self.change_context(id)
      return
    FORMATS = ["DjVuTXT"]
    dir = tempfile.mkdtemp(prefix=self.name)
    ia.download(identifier=id, destdir=dir, no_directory=True, formats=FORMATS)
    if not os.listdir(dir):
      self.console.print("[red]Item was not loaded![/]")
      return
    docs = SimpleDirectoryReader(dir).load_data()
    idx = GPTSimpleVectorIndex.from_documents(docs, service_context=self.sctx)
    ctx = {
      "context": "ia",
      "url": f"https://archive.org/details/{id}",
      "dir": dir,
      "index": idx
    }
    self.add_context(id, ctx)
    self.console.print("[cyan]The item is loaded, ask questions or try the following:[/]")
    self.console.print("[italic dim]Summary[/]")
    self.console.print("[italic dim]Highlights[/]")
    self.console.print("[italic dim]Metadata in JSON[/]")

  def load_tv(self, chan, dt=""):
    self.console.print("[red]Not implemented yet![/]")

  def load_wbm(self, url, dt=""):
    self.console.print("[red]Not implemented yet![/]")

  def load_wiki(self, title, lang="en"):
    self.console.print("[red]Not implemented yet![/]")

  def load_file(self, path):
    self.console.print("[red]Not implemented yet![/]")

  def get_openai_key(self):
    if not os.getenv("OPENAI_API_KEY"):
      self.console.print("[red]Set OpenAI API key obtained from https://beta.openai.com/account/api-keys[/]")
      self.console.print("Usage: [cyan]config openai[/] [magenta]<KEY>[/]")
      return
    key = os.getenv("OPENAI_API_KEY")
    self.console.print(f"[cyan]OPENAI_API_KEY[/]: [magenta]{key}[/]")

  def set_openai_key(self, key):
    os.environ["OPENAI_API_KEY"] = key

  def ask_gpt(self, p):
    if not self.id:
      self.console.print("[red]No context is set, load a new one or change to an existing one![/]")
      return
    idx = self.store[self.id]["index"]
    res = idx.query(p)
    self.queries += 1
    self.tokens += idx.service_context.embed_model.total_tokens_used
    self.console.print(res.response)


def main():
  cp = IaCopilot(name=__NAME, version=__VERSION)
  if {"-h", "--help", "help"}.intersection(sys.argv):
    cp.print_help()
    sys.exit()
  if {"-v", "--version", "version"}.intersection(sys.argv):
    cp.print_version()
    sys.exit()
  readline.parse_and_bind("tab: complete")
  readline.set_completer(cp.tc.completer)
  cp.welcome()
  if not os.getenv("OPENAI_API_KEY"):
    cp.get_openai_key()
  try:
    while True:
      p = input(cp.prompt).strip()
      if not p:
        continue
      try:
        cp.parse(p)
      except pydantic.error_wrappers.ValidationError as e:
        if "OPENAI_API_KEY" in str(e):
          print(e)
          cp.get_openai_key()
        else:
          print(e)
      except openai.error.InvalidRequestError as e:
        print(e)
  except (KeyboardInterrupt, SystemExit) as e:
    cp.console.print("[red]Exiting...[/]")
    sys.exit()


if __name__=="__main__":
  main()
