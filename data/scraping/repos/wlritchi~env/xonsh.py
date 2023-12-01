#!/usr/bin/env python3

import base64
import json
import os
import random
import subprocess
import sys
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import product
from math import log
from os.path import (
    basename,
    dirname,
    exists,
    isabs,
    isdir,
    isfile,
    islink,
    ismount,
    lexists,
    realpath,
    relpath,
    samefile,
)
from random import randint
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    TypeVar,
)

import xonsh
from xonsh.ansi_colors import register_custom_ansi_style
from xonsh.built_ins import XSH
from xonsh.tools import print_color
from xonsh.xontribs import xontribs_load
from xonsh.xoreutils import _which

try:
    import numpy as np
    from numpy.typing import NDArray
except Exception:
    pass

XSH.env['XONSH_SHOW_TRACEBACK'] = True
XSH.env['XONSH_HISTORY_BACKEND'] = 'sqlite'
XSH.env['XONSH_HISTORY_SIZE'] = '1000000 commands'
XSH.env['fzf_history_binding'] = 'c-r'


def _setup():
    def which(bin: str):
        try:
            _which.which(bin)
            return True
        except _which.WhichError:
            return False


    def can_autoinstall():
        return '.local/pipx/venvs/xonsh' in sys.prefix


    def autoinstall(pkgname: str):
        print_color(f"{{BLUE}}↻{{RESET}} xonsh - installing {pkgname}")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkgname], check=True)
            return True
        except subprocess.CalledProcessError:
            print_color(f"{{RED}}🗙{{RESET}} xonsh - failed to install {pkgname}")
            return False


    def has_package(package_import: str):
        # lazy import
        from importlib import import_module
        try:
            import_module(package_import)
            return True
        except ModuleNotFoundError:
            pass
        return False


    def ensure_package(
        missing_package_collector: Set[str],
        package_spec: str | Tuple[str, str],
    ):
        # lazy import
        from importlib import import_module
        if isinstance(package_spec, tuple):
            (package_import, package_pip) = package_spec
        else:
            package_import = package_spec
            package_pip = package_spec.replace('_', '-').replace('.', '-')
        if has_package(package_import):
            return True
        elif can_autoinstall() and autoinstall(package_pip or package_import):
            return True
        missing_package_collector.add(package_pip)
        return False


    # package spec: import name, or tuple (import name, pip name)
    EARLY_PACKAGES = (
        'catppuccin',
        'pygments',
        'prompt_toolkit',
    )
    CONVENIENCE_PACKAGES = (
        'numpy',  # imported as np if available
        'openai',  # used by gpt
        'pytimeparse',  # used by randtimedelta
        'tiktoken',  # used by gpt
        ('skimage', 'scikit-image'),
    )
    # xontrib spec: tuple (binary deps, package spec)
    XONTRIBS = (
        ([], 'xontrib.argcomplete'),
        ([], 'xontrib_avox_poetry'),
        ([], 'xontrib.jedi'),
        ([], 'xontrib.pipeliner'),
        ([], 'xontrib.vox'),
        ([], 'xontrib.whole_word_jumping'),
        (['fzf'], 'xontrib.fzf-widgets'),
        (['zoxide'], 'xontrib.zoxide'),
    )

    def prepare_early_packages() -> bool:
        missing_packages = set()
        for package in EARLY_PACKAGES:
            ensure_package(missing_packages, package)
        if missing_packages:
            print_color(f"{{YELLOW}}⚠{{RESET}} xonsh - missing packages for standard environment (xpip install {' '.join(missing_packages)} to fix)")
            return False
        return True

    def prepare_packages():
        missing_packages = set()
        for package in CONVENIENCE_PACKAGES:
            ensure_package(missing_packages, package)
        for xontrib in XONTRIBS:
            bins, package = xontrib
            has_bins = True
            for binary in bins:
                if not which(binary):
                    has_bins = False
            if has_bins and ensure_package(missing_packages, package):
                if isinstance(package, tuple):
                    package_import, _package_pip = package
                else:
                    package_import = package
                xontribs_load(package_import[8:])

        if missing_packages:
            print_color(f"{{YELLOW}}⚠{{RESET}} xonsh - missing packages for standard environment (xpip install {' '.join(missing_packages)} to fix)")


    def setup_colors():
        if not prepare_early_packages():
            return

        from catppuccin import Flavour
        from pygments.token import Token
        from xonsh.pyghooks import register_custom_pygments_style

        catppuccin_macchiato = Flavour.macchiato()
        color_tokens = {
            getattr(Token.Color, key.upper()): f"#{value.hex}"
            for key, value in catppuccin_macchiato.__dict__.items()
        }
        intense_color_tokens = {
            getattr(Token.Color, f"INTENSE_{key}".upper()): f"#{value.hex}"
            for key, value in catppuccin_macchiato.__dict__.items()
        }
        register_custom_pygments_style(
            'catppuccin-macchiato-term',
            {
                **color_tokens,
                **intense_color_tokens,
                # alias other color names xonsh expects
                Token.Color.PURPLE: f"#{catppuccin_macchiato.pink.hex}",
                Token.Color.INTENSE_PURPLE: f"#{catppuccin_macchiato.pink.hex}",
                Token.Color.CYAN: f"#{catppuccin_macchiato.teal.hex}",
                Token.Color.INTENSE_CYAN: f"#{catppuccin_macchiato.teal.hex}",
                Token.Color.WHITE: f"#{catppuccin_macchiato.subtext0.hex}",
                Token.Color.INTENSE_WHITE: f"#{catppuccin_macchiato.subtext1.hex}",
                Token.Color.BLACK: f"#{catppuccin_macchiato.surface1.hex}",
                Token.Color.INTENSE_BLACK: f"#{catppuccin_macchiato.surface2.hex}",
            },
            base='catppuccin-macchiato',
        )
        XSH.env['XONSH_COLOR_STYLE'] = 'catppuccin-macchiato-term'

    setup_colors()


    GPT_MODEL_CHOICES_BY_TOKEN_COUNT = {
        'gpt-3.5-turbo': [
            (4096, 'gpt-3.5-turbo'),
            (16384, 'gpt-3.5-turbo-16k'),
        ],
        'gpt-4': [
            (8192, 'gpt-4'),
            (32765, 'gpt-4-32k'),
        ],
    }
    GPT_MODEL_PRICING = {  # prompt, completion, per 1000 tokens
        'gpt-3.5-turbo': (0.0015, 0.002),
        'gpt-3.5-turbo-16k': (0.003, 0.004),
        'gpt-4': (0.03, 0.06),
        'gpt-4-32k': (0.06, 0.12),
    }
    GPT_MODEL_EXTRA_TOKENS = {  # per message, per role switch
        'gpt-3.5-turbo': (3, 1),  # used to be (4, 1) in the gpt-3.5-turbo-0301 model
        'gpt-4': (3, 1),
    }

    GPT_STREAMING = True

    gpt_cost_acc = 0
    gpt_messages = []
    gpt_tokens = 0

    def _query_gpt(query, flavor):
        nonlocal gpt_cost_acc, gpt_messages, gpt_tokens

        try:
            import openai
        except ModuleNotFoundError:
            print("Unable to load openai module, cannot query ChatGPT", file=sys.stderr)
            return 1
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model(flavor)
        except ModuleNotFoundError:
            print("Warning: Unable to load tiktoken module, cannot estimate token usage", file=sys.stderr)
            encoder = None

        # cheapo bare words approximation
        if len(query) > 1:
            query_str = ' '.join(f'"{q}"' if ' ' in q else q for q in query)
        else:
            query_str = query[0]

        prompt_tokens = 0
        if encoder is not None:
            tokens_per_message, tokens_per_role_switch = GPT_MODEL_EXTRA_TOKENS[flavor]
            extra_tokens = tokens_per_message * 2 + tokens_per_role_switch
            if gpt_messages:
                extra_tokens += tokens_per_role_switch
            prompt_tokens = len(encoder.encode(query_str)) + extra_tokens
        total_tokens = gpt_tokens + prompt_tokens

        model_choices = GPT_MODEL_CHOICES_BY_TOKEN_COUNT[flavor]
        for max_tokens, model in model_choices:
            if total_tokens < max_tokens:
                break

        print(f'[{model}]')

        gpt_messages.append({
            'role': 'user',
            'content': query_str,
        })

        response = openai.ChatCompletion.create(
            model=model,
            messages=gpt_messages,
            stream=GPT_STREAMING,
        )

        if GPT_STREAMING:
            response_message = {}
            for chunk in response:
                chunk_delta = chunk['choices'][0]['delta']
                if 'role' in chunk_delta:
                    response_message['role'] = chunk_delta['role']
                if 'content' in chunk_delta:
                    print(chunk_delta['content'], end='')
                    response_message['content'] = response_message.get('content', '') + chunk_delta['content']
            print()
            gpt_messages.append(response_message)

            if encoder is not None:
                completion_tokens = len(encoder.encode(response_message['content']))
                gpt_tokens += prompt_tokens + completion_tokens
        else:
            response_message = response['choices'][0]['message']

            print(response_message['content'])
            gpt_messages.append(response_message)

            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            gpt_tokens = response['usage']['total_tokens']

        prompt_price, completion_price = GPT_MODEL_PRICING[model]
        gpt_cost_acc += (prompt_price * prompt_tokens + completion_price * completion_tokens) / 1000

    def _gpt(query):
        _query_gpt(query, 'gpt-3.5-turbo')

    def _gpt4(query):
        _query_gpt(query, 'gpt-4')

    XSH.aliases['gpt'] = _gpt
    XSH.aliases['gpt4'] = _gpt4


    # set up prompt
    def _prompt():
        global _
        nonlocal gpt_cost_acc, gpt_tokens
        rtn_str = ''
        try:
            if _.rtn != 0:
                rtn_str = '{RED}' + f'[{_.rtn}]'
        except AttributeError: # previous command has no return code (e.g. because it's a xonsh function)
            pass
        except NameError: # no _, no previous command
            pass
        gpt_cost_str = f'{{BLUE}}{gpt_cost_acc:.2f}|{gpt_tokens})' if gpt_cost_acc else ''
        rtn_formatted = '\n' + gpt_cost_str + rtn_str
        return rtn_formatted + '{YELLOW}{localtime}{GREEN}{user}@{hostname}{BLUE}{cwd}{YELLOW}{curr_branch:({})}{RESET}$ '


    XSH.env['PROMPT'] = _prompt


    def prepare_aliases():
        # use aliases to resolve naming conflicts and overwrite default behaviour

        XSH.aliases['gap'] = 'git add -p'  # some algebra package
        XSH.aliases['gm'] = 'git merge'  # graphicsmagick
        XSH.aliases['gs'] = 'git status'  # ghostscript

        if which('grmx'):  # macos, with brew: gnu rm
            XSH.aliases['grm'] = 'grmx'

        if which('bat'):
            XSH.aliases['cat'] = 'bat'
        if which('eza'):
            XSH.aliases['ls'] = 'eza'
        elif which('exa'):
            XSH.aliases['ls'] = 'exa'

        if which('dd-shim'):
            XSH.aliases['dd'] = 'dd-shim'
        if which('gradle-shim'):
            XSH.aliases['gradle'] = 'gradle-shim'
        if which('yay-shim'):
            XSH.aliases['yay'] = 'yay-shim'

        if which('fluxx'):
            XSH.aliases['flux'] = 'fluxx'
        if which('helmx'):
            XSH.aliases['helm'] = 'helmx'
        if which('fluxx'):
            XSH.aliases['kubectl'] = 'kubectlx'

        if which('sshx'):
            XSH.aliases['ssh'] = 'sshx'
        if which('sshfsx'):
            XSH.aliases['sshfs'] = 'sshfsx'
        if which('moshx'):
            XSH.aliases['mosh'] = 'moshx'

        # xonsh-only, workaround for lack of ergonomic "time" builtin
        if which('timex'):
            XSH.aliases['time'] = 'timex'

        def _cd(args):
            if len(args) > 0:
                _r = xonsh.dirstack.pushd(args)
                if _r[1] is not None:
                    print(_r[1].strip(), file=sys.stderr)
                return _r[2]
            else:
                xonsh.dirstack.popd(args)
        XSH.aliases['cd'] = _cd

        def _mkcd(args):
            if len(args) != 1:
                print('Usage: mkcd DIRECTORY', file=sys.stderr)
                return 1
            dir = args[0]
            os.mkdir(dir)
            xonsh.dirstack.pushd([dir])
        XSH.aliases['mkcd'] = _mkcd

        # # temporary workaround for xonsh bug in 0.9.27
        # # see https://github.com/xonsh/xonsh/issues/4243 and https://github.com/xonsh/xonsh/issues/2404
        # XSH.aliases['gs'] = '$[git status]'
        # def _gd(args):
        #     $[git diff @(args)]
        # XSH.aliases['gd'] = _gd
        # def _glog(args):
        #     $[~/.wlrenv/bin/aliases/glog @(args)]
        # XSH.aliases['glog'] = _glog
        # def _gtree(args):
        #     $[~/.wlrenv/bin/aliases/gtree @(args)]
        # XSH.aliases['gtree'] = _gtree

        def _source(source_fn):
            """Wrap the source alias to handle attempts to activate a venv.

            Some tools, such as VS Code, run a shell and type
                source <path>/bin/activate
            into that shell, in order for the shell to run in the venv.
            Unfortunately, xonsh does not play well with standard venv activation
            scripts. Instead, xonsh provides the vox xontrib, loaded above, which
            offers similar functionality. This wrapper catches attepts to source venv
            activation scripts (which wouldn't work anyway, as xonsh's source expects
            only xonsh-flavoured inputs), and converts them into calls to vox."""

            def wrapper(args):
                if len(args) == 1 and args[0].endswith('/bin/activate'):
                    virtualenv_name = args[0][:-13]
                    from xontrib.voxapi import Vox
                    Vox().activate(virtualenv_name)
                else:
                    source_fn(args)

            return wrapper
        XSH.aliases['source'] = _source(XSH.aliases['source'])


    def late_init():
        prepare_aliases()
        prepare_packages()


    threading.Thread(target=late_init).start()


_setup()
del _setup


def coin():
    return 'heads' if randint(0, 1) else 'tails'


def ndm(n=1, m=6):
    return sum(randint(1, m) for _ in range(n))


def d4(n=1):
    return ndm(n, 4)


def d6(n=1):
    return ndm(n, 6)


def d8(n=1):
    return ndm(n, 8)


def d20(n=1):
    return ndm(n, 20)


def shuffle(items):
    l = list(items)
    random.shuffle(l)
    return l


def choose(items):
    l = list(items)
    return l[randint(0, len(l) - 1)]


def parsetimedelta(x):
    if isinstance(x, str):
        from pytimeparse.timeparse import timeparse
        x = timeparse(x)
    if isinstance(x, int) or isinstance(x, float):
        x = timedelta(seconds=x)
    if not isinstance(x, timedelta):
        raise ValueError(f"Expected string, number of seconds, or timedelta instance; got {timedelta}")
    return x


def randtimedelta(a, b=None):
    if b is None:
        a, b = (timedelta(0), a)
    a = parsetimedelta(a)
    b = parsetimedelta(b)
    seconds = randint(int(a.total_seconds()), int(b.total_seconds()))
    return str(timedelta(seconds=seconds))


def snap_to_grid(point, grid_spacing=10, grid_reference=0):
    return grid_reference + grid_spacing * round((point - grid_reference) / grid_spacing)


def bits(n):
    # https://stackoverflow.com/a/4859937
    if isinstance(n, str):
        n = int(n, 16)
    return bin(n)[2:].zfill(8)


def lines(file):
    with open(file, 'r') as f:
        return [x.strip() for x in f.readlines()]
