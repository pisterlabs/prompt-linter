from collections import ChainMap
from fractions import Fraction

from ._base_unit_map import base_unit_map
from ._coherent_unit_map import coherent_unit_map
from .base import kg
from .._prefixes import _prefix_many_to_many
from .._utils import names_and_abbrevs, register_unit_map


gramdef = {names_and_abbrevs('gram'): (kg * Fraction(1, 1000)).set_name('g')}

register_unit_map(gramdef).register_many(ignore_if_exists=True, **_prefix_many_to_many(
    ChainMap(base_unit_map, coherent_unit_map, gramdef), except_=('kg',)))
