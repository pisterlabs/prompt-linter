#!/usr/bin/env python

# This file is part of Lifewatch DAAP.
# Copyright (C) 2015 Ana Yaiza Rodriguez Marrero.
#
# Lifewatch DAAP is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lifewatch DAAP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Lifewatch DAAP. If not, see <http://www.gnu.org/licenses/>.

# This file is part of Invenio.
# Copyright (C) 2013 CERN.
##
# Invenio is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
##
# Invenio is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with Invenio; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.

"""
Simple tasklet that is called after a bibupload of an updated record.
"""

from invenio.modules.pidstore.tasks import datacite_update

# from zenodo.modules.deposit.tasks import openaire_altmetric_update
# from zenodo.modules.preservationmeter.tasks \
#    import calculate_preservation_score


def bst_openaire_update_upload(recid=None):
    """Tasklet to run after a new record has been uploaded."""
    if recid is None:
        return

    # Ship of tasks to Celery for background processing
    datacite_update.delay(recid)
    # openaire_altmetric_update.delay([recid])
    # calculate_preservation_score.delay(recid=recid)


if __name__ == '__main__':
    bst_openaire_update_upload()
