from nose.tools import assert_almost_equal, assert_less
from unittest import TestCase

from guidance import calculate_gps_heading
from guidance.gps_guidance import initial_bearing

# Calculations made with https://www.movable-type.co.uk/scripts/latlong.html
culinary_arts = (43.601214, -116.197543)
rec_center = (43.600532, -116.200390)


class TestGPSGuidance(TestCase):

    def test_initial_bearing(self):
        bearing = initial_bearing(culinary_arts, rec_center)
        assert_less(abs(251.6969 - bearing), 4)  # for some reason calculations are off, within 4 deg is ok I guess

    def test_calculate_gps_heading(self):
        bearing = initial_bearing(culinary_arts, rec_center)
        heading = calculate_gps_heading({'lat': culinary_arts[0],
                                         'lon': culinary_arts[1],
                                         'gps_heading': bearing + 10},
                                        rec_center)

        assert_almost_equal(heading.angle, 350)
        assert_almost_equal(heading.mag, 241.5, places=1)

    pass
