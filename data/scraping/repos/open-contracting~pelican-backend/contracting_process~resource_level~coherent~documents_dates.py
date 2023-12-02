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

    for prefix in (
        "planning",
        "tender",
        "awards",
        "contracts",
        "contracts.implementation",
    ):
        for first_path, second_path, split in (
            (f"{prefix}.documents.datePublished", f"{prefix}.documents.dateModified", True),
            (f"{prefix}.documents.datePublished", "date", False),
            (f"{prefix}.documents.dateModified", "date", False),
        ):
            first_dates = _get_values(first_path)
            second_dates = _get_values(second_path)
            pairs.extend(
                (first_date, second_date)
                for first_date in first_dates
                for second_date in second_dates
                if not split or first_date["path"].rsplit(".", 1)[0] == second_date["path"].rsplit(".", 1)[0]
            )

    return coherent_dates_check(version, pairs)
