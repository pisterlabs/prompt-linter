
import os
import sys
import json
import argparse

from ..config import version
from ..architecture.base import CognitiveArchitecture
from ..architecture.orchestrator import Serial, Async
from ..architecture.utility import PromptTee

from ..lm import OpenAI, TfLM, Llama
LMs = { 'OpenAI' : OpenAI, 'TfLM' : TfLM, 'LLaMa' : Llama }

def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version',  action='version', version=f'AutoCog {version}') # TODO `autocog.version:str=read('VERSION')`

    parser.add_argument('--orch',     help="""Type of orchestrator: `serial` or `async`.""", default='serial')

    parser.add_argument('--lm',       action='append', help="""Inlined JSON or path to a JSON file: `{ "text"   : { "cls" : "OpenAI", ... } }` see TODO for details.""")
    parser.add_argument('--program',  action='append', help="""Inlined JSON or path to a JSON file: `{ "writer" : { "filepath" : "./library/writer/simple.sta", ... } }` see TODO for details.""")
    parser.add_argument('--tool',     action='append', help="""Inlined JSON or path to a JSON file: `{ "search" : { "cls" : "SerpApi", ... } }` see TODO for details.""")

    parser.add_argument('--prefix',   help="""String to identify this instance of AutoCog (used when displaying and saving the prompts)""", default='autocog')
    parser.add_argument('--tee',      help="""Filepath or `stdout` or `stderr`. If present, prompts will be append to that file as they are executed.""")
    parser.add_argument('--fmt',      help="""Format string used to save individual prompts to files. If present but empty (or `default`), `{p}/{c}/{t}-{i}.txt` is used. `p` is the prefix. `c` is the sequence id of the call. `t` is the prompt name. `i` is the prompt sequence id. WARNING! This will change as the schema is obsolete!""")

    parser.add_argument('--serve',    help="""Whether to launch the flask server.""", action='store_true')
    parser.add_argument('--host',     help="""Host for flask server.""", default='localhost')
    parser.add_argument('--port',     help="""Port for flask server.""", default='5000')
    parser.add_argument('--debug',    help="""Whether to run the flask server in debug mode.""", action='store_true')


    parser.add_argument('--command',  action='append', help="""Inlined JSON or path to a JSON file: `{ 'callee' : 'writer',  ... }` see TODO for details.""")
    parser.add_argument('--opath',    help="""Directory where results are stored.""", default=os.getcwd())
    
    return parser

def parse_json(arg):
    if os.path.exists(arg):
        return json.load(open(arg))
    else:
        return json.loads(arg)

def parse_lm(cls:str, **kwargs):
    global LMs
    if cls in LMs:
        cls = LMs[cls]
    else:
        raise Exception(f"Unknown LM class: {cls} (should be one of {','.join(LMs.keys())})")

    model_kwargs = kwargs['model'] if 'model' in kwargs else {}
    model_kwargs = cls.create(**model_kwargs)
    if 'config' in kwargs:
        model_kwargs.update({ "completion_kwargs" : kwargs['config'] })
    return cls(**model_kwargs)


def parse_lms(lms):
    return { fmt : parse_lm(**lm) for (fmt,lm) in lms.items() }

def parseargs(argv):
    parser = argparser()
    args = parser.parse_args(argv)

    pipe_kwargs = { 'prefix' : args.prefix }
    if args.tee is not None:
        if args.tee == 'stdout':
            pipe_kwargs.update({ 'tee' : sys.stdout })
        elif args.tee == 'stderr':
            pipe_kwargs.update({ 'tee' : sys.stderr })
        else:
            pipe_kwargs.update({ 'tee' : open(args.tee,'w') })

    if args.fmt is not None:
        if args.fmt == '' or args.fmt == 'default':
            pipe_kwargs.update({ 'fmt' : '{p}/{c}/{t}-{i}.txt' })
        else:
            pipe_kwargs.update({ 'fmt' : args.fmt })

    if args.orch == 'serial':
        Orch = Serial
    elif args.orch == 'async':
        Orch = Async
    else:
        raise Exception(f"Unknown Orchestrator: {args.orch}")
    arch = CognitiveArchitecture(Orch=Orch, pipe=PromptTee(**pipe_kwargs))

    if args.lm is not None:
        for lm in args.lm:
            arch.orchestrator.LMs.update(parse_lms(parse_json(lm)))

    programs = {}
    if args.program is not None:
        for prog in args.program:
            programs.update(parse_json(prog))
    for (tag,program) in programs.items():
        arch.load(tag=tag, **program)

    tools = {}
    if args.tool is not None:
        for tool in args.tool:
            tools.update(parse_json(tool))
    for (tag,tool) in tools.items():
        raise NotImplementedError()

    return {
        'arch' : arch,  'serve' : args.serve, 'opath' : args.opath,
        'host' : args.host, 'port' : int(args.port), 'debug' : args.debug,
        'commands' : None if args.command is None else [ parse_json(cmd) for cmd in args.command ]
    }
