#!/usr/bin/env python3
"""
Author : Xinyuan Chen <45612704+tddschn@users.noreply.github.com>
Date   : 2023-04-17
Purpose: Fill the commands help message for the README
"""

import argparse
from pathlib import Path
from langchain_utils.config import (
    _README_PATH,
    _README_COMMANDS,
    _README_TEMPLATE,
    _COMMAND_USAGE_TEMPLATE,
)
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jinja2 import Template


def get_command_usage(command: str) -> str:
    import subprocess

    # run `command --help` and get the output
    p = subprocess.run([command, '--help'], capture_output=True)
    help_message = p.stdout.decode('utf-8')
    return help_message


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Fill the commands help message for the README',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return parser.parse_args()


def main():
    """Make a jazz noise here"""

    args = get_args()
    prompt_commands_and_usages_parts = []

    from jinja2 import Template

    command_usage_jinja_template: Template = Template(
        _COMMAND_USAGE_TEMPLATE.read_text()
    )
    readme_jinja_template = Template(_README_TEMPLATE.read_text())
    for command in _README_COMMANDS:
        command_usage = get_command_usage(command)
        rendered_command_usage = command_usage_jinja_template.render(
            command=command, command_usage=command_usage
        )
        prompt_commands_and_usages_parts.append(rendered_command_usage)
    rendered_readme = readme_jinja_template.render(
        prompt_commands_and_usages='\n'.join(prompt_commands_and_usages_parts)
    )
    # write back to README.md
    _README_PATH.write_text(rendered_readme)


if __name__ == '__main__':
    main()
