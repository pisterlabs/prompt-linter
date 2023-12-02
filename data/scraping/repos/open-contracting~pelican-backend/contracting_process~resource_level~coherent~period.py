"""
.. seealso::

   :func:`pelican.util.checks.coherent_dates_check
"""

from pelican.util.checks import coherent_dates_check
from pelican.util.getter import get_values

version = 1.0


def calculate(item):
    pairs = []

    for first_path, second_path, split in (
        ("tender.tenderPeriod.startDate", "tender.tenderPeriod.endDate", False),
        ("tender.enquiryPeriod.startDate", "tender.enquiryPeriod.endDate", False),
        ("tender.awardPeriod.startDate", "tender.awardPeriod.endDate", False),
        ("tender.contractPeriod.startDate", "tender.contractPeriod.endDate", False),
        ("awards.contractPeriod.startDate", "awards.contractPeriod.endDate", True),
        ("contracts.period.startDate", "contracts.period.endDate", True),
    ):
        first_dates = get_values(item, first_path)
        second_dates = get_values(item, second_path)
        pairs.extend(
            (first_date, second_date)
            for first_date in first_dates
            for second_date in second_dates
            if not split or first_date["path"].split(".", 1)[0] == second_date["path"].split(".", 1)[0]
        )

    return coherent_dates_check(version, pairs)
