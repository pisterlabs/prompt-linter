"""
.. seealso::

   :func:`pelican.util.checks.coherent_dates_check
"""

from functools import lru_cache

from pelican.util.checks import coherent_dates_check
from pelican.util.getter import get_values

version = 1.0


def calculate(item):
    @lru_cache
    def _get_values(path):
        return get_values(item, path)

    pairs = []

    for first_path, second_path, split in (
        ("tender.tenderPeriod.startDate", "tender.amendments.date", False),
        ("awards.date", "awards.amendments.date", True),
        ("contracts.dateSigned", "contracts.amendments.date", True),
        ("tender.amendments.date", "date", False),
        ("awards.amendments.date", "date", False),
        ("contracts.amendments.date", "date", False),
    ):
        first_dates = _get_values(first_path)
        second_dates = _get_values(second_path)
        pairs.extend(
            (first_date, second_date)
            for first_date in first_dates
            for second_date in second_dates
            if not split or first_date["path"].split(".", 1)[0] == second_date["path"].split(".", 1)[0]
        )

    return coherent_dates_check(version, pairs)
