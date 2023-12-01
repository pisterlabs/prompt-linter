# -*- coding: utf-8 -*-
#
# This file is part of Invenio.
# Copyright (C) 2015-2019 CERN.
#
# Invenio is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.

"""OpenAIRE service integration for Invenio repositories."""

from __future__ import absolute_import, print_function

from copy import deepcopy

from celery import chain, shared_task
from flask import current_app
from invenio_db import db
from invenio_indexer.api import RecordIndexer
from invenio_pidstore.errors import PIDDoesNotExistError
from invenio_pidstore.resolver import Resolver
from invenio_records.api import Record

from .loaders import LocalFundRefLoader, LocalOAIRELoader, \
    RemoteFundRefLoader, RemoteOAIRELoader
from .minters import funder_minter, grant_minter


@shared_task(ignore_result=True)
def harvest_fundref(source=None):
    """Harvest funders from FundRef and store as authority records."""
    loader = LocalFundRefLoader(source=source) if source \
        else RemoteFundRefLoader()
    for funder_json in loader.iter_funders():
        register_funder.delay(funder_json)


@shared_task(ignore_result=True)
def harvest_openaire_projects(source=None, setspec=None):
    """Harvest grants from OpenAIRE and store as authority records."""
    loader = LocalOAIRELoader(source=source) if source \
        else RemoteOAIRELoader(setspec=setspec)
    for grant_json in loader.iter_grants():
        register_grant.delay(grant_json)


@shared_task(ignore_result=True)
def harvest_all_openaire_projects():
    """Reharvest all grants from OpenAIRE.

    Harvest all OpenAIRE grants in a chain to prevent OpenAIRE
    overloading from multiple parallel harvesting.
    """
    setspecs = current_app.config['OPENAIRE_GRANTS_SPECS']
    chain(harvest_openaire_projects.s(setspec=setspec)
          for setspec in setspecs).apply_async()


@shared_task(ignore_result=True)
def register_funder(data):
    """Register the funder JSON in records and create a PID."""
    create_or_update_record(data, 'frdoi', 'doi', funder_minter)


@shared_task(ignore_result=True, rate_limit='20/s')
def register_grant(data):
    """Register the grant JSON in records and create a PID."""
    create_or_update_record(data, 'grant', 'internal_id', grant_minter)


def create_or_update_record(data, pid_type, id_key, minter):
    """Register a funder or grant."""
    resolver = Resolver(
        pid_type=pid_type, object_type='rec', getter=Record.get_record)

    try:
        pid, record = resolver.resolve(data[id_key])
        data_c = deepcopy(data)
        del data_c['remote_modified']
        record_c = deepcopy(record)
        del record_c['remote_modified']
        # All grants on OpenAIRE are modified periodically even if nothing
        # has changed. We need to check for actual differences in the metadata
        if data_c != record_c:
            record.update(data)
            record.commit()
            record_id = record.id
            db.session.commit()
            RecordIndexer().index_by_id(str(record_id))
    except PIDDoesNotExistError:
        record = Record.create(data)
        record_id = record.id
        minter(record.id, data)
        db.session.commit()
        RecordIndexer().index_by_id(str(record_id))
