"""
.. seealso::

   :func:`pelican.util.checks.coherent_dates_check
"""

from functools import lru_cache

from pelican.util.checks import coherent_dates_check
from pelican.util.getter import deep_has, get_values

version = 1.0


def calculate(item):
    @lru_cache
    def _get_values(path):
        return get_values(item, path)

    pairs = []

    for first_path, second_path, split in (
        ("tender.tenderPeriod.endDate", "tender.contractPeriod.startDate", False),
        ("tender.tenderPeriod.endDate", "awards.date", False),
        ("tender.tenderPeriod.endDate", "contracts.dateSigned", False),
        ("awards.date", "date", False),
        ("contracts.dateSigned", "date", False),
        ("contracts.implementation.transactions.date", "date", False),
        ("contracts.dateSigned", "contracts.implementation.transactions.date", True),
    ):
        first_dates = _get_values(first_path)
        second_dates = _get_values(second_path)
        pairs.extend(
            (first_date, second_date)
            for first_date in first_dates
            for second_date in second_dates
            if not split or first_date["path"].split(".", 1)[0] == second_date["path"].split(".", 1)[0]
        )

    awards = get_values(item, "awards")
    contracts = get_values(item, "contracts")
    pairs.extend(
        (
            {"path": f"{award['path']}.date", "value": award["value"]["date"]},
            {"path": f"{contract['path']}.dateSigned", "value": contract["value"]["dateSigned"]},
        )
        for award in awards
        if deep_has(award["value"], "date") and deep_has(award["value"], "id")
        for contract in contracts
        if deep_has(contract["value"], "dateSigned")
        and deep_has(contract["value"], "awardID")
        and str(award["value"]["id"]) == str(contract["value"]["awardID"])
    )

    return coherent_dates_check(version, pairs)
