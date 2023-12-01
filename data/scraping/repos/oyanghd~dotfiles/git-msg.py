#!/usr/bin/env python3

import os
import sys
import configparser
import subprocess

import openai

CONFIG_FILE = os.path.expanduser('~/.krrc')


def get_diff():
    try:
        opt = sys.argv[1]
    except IndexError:
        opt = None

    opts = [None, '--cached', 'HEAD', 'HEAD~']
    try:
        opts.remove(opt)
    except ValueError:
        pass

    for opt in (opt, *opts):
        cmd = ['git', 'diff']
        if opt:
            cmd.append(opt)
        out = subprocess.check_output(cmd, text=True).strip()
        if out:
            return out


def main():
    if not os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding='utf-8') as fp:
            fp.write('[openai]\nsecret_key=\n')

        print('OpenAI API config file created at',
              CONFIG_FILE, file=sys.stderr)
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    openai.api_key = config['openai']['secret_key'].strip('"').strip("'")
    openai.proxy = 'http://127.0.0.1:10808'

    try:
        sys.argv.remove('-l')
    except ValueError:
        long = False
    else:
        long = True

    try:
        sys.argv.remove('-z')
    except ValueError:
        lang = ''
    else:
        lang = ' in Simplified Chinese'

    diff = get_diff()
    if not diff:
        print('No diff.', file=sys.stderr)
        return
    print(diff)

    if long:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant writing git commit messages.'},
            {'role': 'user', 'content': diff +
                f'\n\nWrite a short commit message and some decriptions{lang}.'},
        ]
    else:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant writing short git commit messages.'},
            {'role': 'user', 'content': diff +
                f'\n\nWrite a short commit message{lang}.'},
        ]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
    )

    for c in response['choices']:
        print(c['message']['content'])


if __name__ == '__main__':
    main()
