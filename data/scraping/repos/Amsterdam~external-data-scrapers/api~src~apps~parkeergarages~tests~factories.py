import factory
from django.contrib.gis.geos import Point
from django.utils import timezone
from factory import fuzzy

from ..models import GuidanceSign, ParkingGuidanceDisplay, ParkingLocation

# Amsterdam.
BBOX = [52.03560, 4.58565, 52.48769, 5.31360]


def get_puntje():
    lat = fuzzy.FuzzyFloat(BBOX[0], BBOX[2]).fuzz()
    lon = fuzzy.FuzzyFloat(BBOX[1], BBOX[3]).fuzz()
    return Point(float(lat), float(lon))


class ParkingLocationFactory(factory.DjangoModelFactory):
    class Meta:
        model = ParkingLocation

    id = factory.Sequence(lambda n: n)
    api_id = fuzzy.FuzzyText(length=4)
    stadsdeel = fuzzy.FuzzyChoice(choices=['A', 'B', 'C'])
    geometrie = get_puntje()
    name = fuzzy.FuzzyText(length=4)
    type = fuzzy.FuzzyText(length=4)
    state = fuzzy.FuzzyText(length=4)
    pub_date = fuzzy.FuzzyDateTime(start_dt=timezone.now())
    free_space_short = fuzzy.FuzzyInteger(0, 100)
    free_space_long = fuzzy.FuzzyInteger(0, 100)
    short_capacity = fuzzy.FuzzyInteger(0, 100)
    long_capacity = fuzzy.FuzzyInteger(0, 100)
    buurt_code = fuzzy.FuzzyText(length=4)
    scraped_at = fuzzy.FuzzyDateTime(start_dt=timezone.now())


class GuidanceSignFactory(factory.DjangoModelFactory):
    class Meta:
        model = GuidanceSign

    id = factory.Sequence(lambda n: n)
    api_id = factory.Sequence(lambda n: str(n))
    name = fuzzy.FuzzyText(length=4)
    state = fuzzy.FuzzyText(length=4)
    type = fuzzy.FuzzyText(length=4)
    stadsdeel = fuzzy.FuzzyChoice(choices=['A', 'B', 'C'])
    geometrie = get_puntje()
    pub_date = fuzzy.FuzzyDateTime(start_dt=timezone.now())
    buurt_code = fuzzy.FuzzyText(length=4)
    scraped_at = fuzzy.FuzzyDateTime(start_dt=timezone.now())


class ParkingGuidanceDisplayFactory(factory.DjangoModelFactory):
    class Meta:
        model = ParkingGuidanceDisplay

    id = factory.Sequence(lambda n: n)
    api_id = factory.Sequence(lambda n: str(n))
    pub_date = fuzzy.FuzzyDateTime(start_dt=timezone.now())
    guidance_sign = factory.SubFactory(GuidanceSignFactory)
    scraped_at = fuzzy.FuzzyDateTime(start_dt=timezone.now())
    output = fuzzy.FuzzyText(length=100)
    output_description = fuzzy.FuzzyText(length=100)
    type = fuzzy.FuzzyText(length=4)
    description = fuzzy.FuzzyText(length=100)
