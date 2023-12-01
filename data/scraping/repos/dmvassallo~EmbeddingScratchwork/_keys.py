"""
API key helpers.

This enables the ``embed`` module to have an ``api_key`` property that, when
set, sets ``openai.api_key``. This is useful because code that consumes this
project's ``embed`` module shouldn't have to use ``openai`` directly or know
about ``openai.api_key``. (Setting ``openai.api_key`` to use ``requests``-based
functions, which don't use ``openai``, would be especially unintuitive.)

Code within the ``embed`` module itself may access ``_keys.api_key``.
"""

__all__ = ['api_key', 'initialize']

import logging
import os
from pathlib import Path
import re
import sys
import types
from typing import Any

import dulwich.repo
import openai

_API_KEY_REGEX = re.compile(r'sk-\w+')
"""Regular expression to check if file contents are an API key."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this submodule (``embed._keys``)."""

api_key: Any = None
"""OpenAI API key. This should only be accessed from ``__init__.py``."""


def initialize(module_or_name):
    """
    Give the module an ``api_key`` property and set it from the environment.

    Setting the property sets ``openai.api_key`` (including this first time).
    """
    if isinstance(module_or_name, str):  # Because no match-case before 3.10.
        module = sys.modules[module_or_name]
    elif isinstance(module_or_name, types.ModuleType):
        module = module_or_name
    else:
        raise TypeError(f'module_or_name is {type(module_or_name).__name__!r}')

    # Give the module an api_key property that updates openai.api_key when set.
    module.__class__ = _ModuleWithApiKeyProperty

    # Try to set the property from the environment or a key file.
    module.api_key = _get_key_if_available()


class _ModuleWithApiKeyProperty(types.ModuleType):
    """A module whose ``api_key`` property also sets ``openai.api_key``."""

    @property
    def api_key(self):
        """OpenAI API key."""
        return api_key

    @api_key.setter
    def api_key(self, value):
        # We really do want to write this submodule's api_key attribute.
        # pylint: disable-next=global-statement
        global api_key
        api_key = openai.api_key = value


def _get_key_if_available():
    """
    Get a reasonable initial value for the ``api_key`` property.

    This checks for the ``OPENAI_API_KEY`` variable, an ``.api_key`` file in
    the current directory, and ``.api_key`` files in higher directories within
    the same Git repository (if the current directory is inside a Git
    repository), in that order. If no key is found, ``None`` is returned.
    """
    if key := os.getenv('OPENAI_API_KEY', '').rstrip():
        _logger.info('API key found in OPENAI_API_KEY environment variable')
        return key

    _logger.debug('API key not found in OPENAI_API_KEY environment variable')

    if key := _read_key_from_file(Path().absolute()):
        return key

    if key := _read_key_from_ancestors_within_repo():
        return key

    _logger.info('API key not set automatically (no key found)')
    return None


def _read_key_from_file(directory):
    """Try to read an API key from an ``.api_key`` file in ``dir``."""
    try:
        key = (directory / '.api_key').read_text(encoding='utf-8').rstrip()
    except FileNotFoundError:
        _logger.debug('No API key file in: %s', directory)
        return None
    except OSError as error:
        _logger.warning('%s', error)
        return None

    if _API_KEY_REGEX.fullmatch(key):
        _logger.info('API key read from file in: %s', directory)
        return key

    _logger.warning('Malformed API key file in: %s', directory)
    return None


def _read_key_from_ancestors_within_repo():
    """Try to read an API key in higher directories inside a Git repository."""
    try:
        repo = dulwich.repo.Repo.discover()
    except dulwich.repo.NotGitRepository:
        _logger.debug('Not in Git repository, stopping key file search')
        return None

    repo_directory = Path(repo.path).absolute()
    directory = Path().absolute()

    if directory == repo_directory:
        return None  # Searching this directory itself is done separately.

    if repo_directory not in directory.parents:
        _logger.error('Git repo status unclear for directory: %s', directory)
        return None

    while directory != repo_directory:
        directory = directory.parent
        if key := _read_key_from_file(directory):
            return key

    return None
