import textwrap

import quantities as pq

from praxes.lib.decorators import memoize
from praxes.physref.lib.mapping import Mapping


class Element(Mapping):

    """
    """

    __slots__ = ['__db', '_symbol', '_keys']

    def _get_data(self, id):
        cursor = self.__db.cursor()
        result = cursor.execute('''select %s from elements
            where element=?''' % id, (self._symbol, )
            ).fetchone()
        assert result is not None
        return result[0]

    @property
    @memoize
    def atomic_mass(self):
        "The average atomic mass of the element, in atomic mass units"
        return (self.molar_mass / pq.constants.N_A).rescale('amu')

    @property
    @memoize
    def atomic_number(self):
        "The atomic number"
        return self._get_data('atomic_number')

    @property
    @memoize
    def _coherent_scattering(self):
        from .scattering import CoherentScattering
        return CoherentScattering(self._symbol, self.__db)

    @property
    @memoize
    def _incoherent_scattering(self):
        from .scattering import IncoherentScattering
        return IncoherentScattering(self._symbol, self.__db)

    @Mapping._keys.getter
    @memoize
    def _keys(self):
        cursor = self.__db.cursor()
        res = cursor.execute(
            '''select iupac_symbol from xray_levels where element=?
            order by absorption_edge desc''',
            (self._symbol,)
            )
        return tuple(i[0] for i in res)

    @property
    @memoize
    def mass_density(self):
        """
        The theoretical solid mass density at standard temperature and
        pressure, regardless of state, in g/cm^3.
        """
        return self._get_data('density') * pq.g / pq.cm**3

    @property
    @memoize
    def molar_mass(self):
        "The molar mass of the element"
        return self._get_data('molar_mass') * pq.g / pq.mol

    @property
    @memoize
    def _photoabsorption(self):
        from .photoabsorption import Photoabsorption
        return Photoabsorption(self._symbol, self.__db)

    @property
    def symbol(self):
        return self._symbol

    def __init__(self, symbol, db):
        """
        symbol is a string, like 'Ca' or 'S'
        """
        self.__db = db
        self._symbol = symbol

    def __getitem__(self, item):
        if not item in self:
            raise KeyError('x-ray level "%s" not recognized' % item)
        from .xraylevel import XrayLevel
        return XrayLevel(self._symbol, item, self.__db)

    def __hash__(self):
        return hash((type(self), self._symbol))

    @memoize
    def __repr__(self):
        return "<Element(%s)>" % self.symbol

    @memoize
    def __str__(self):
        return textwrap.dedent(
            """\
            Element(%s)
              mass density: %s
              molar mass: %s
              x-ray levels: %s""" % (
                self.symbol,
                self.mass_density,
                self.molar_mass,
                self.keys()
                )
            )

    def photoabsorption_cross_section(self, energy, mass=True):
        """
        Return the photoabsorption cross section as a function of
        energy. The energy must be within the range 100 < E < 8e5 eV.

        Cross-sections at energies below 250 eV should not be considered
        reliable.

        If *mass* is True, return the cross-section per gram in cm^2/g.
        If *mass* is False, return the cross-section per atom in cm^2.
        """
        res = self._photoabsorption(energy)
        if not mass:
            res *= self.atomic_mass.rescale('g')
        return res

    def coherent_scattering_cross_section(self, energy, mass=True):
        """
        Return the coherent-scattering cross section in cm^2/g as a function of
        energy. The energy must be within the range 100 < E < 8e5 eV.

        Cross-sections at energies below 250 eV should not be considered
        reliable.

        If *mass* is True, return the cross-section per gram in cm^2/g.
        If *mass* is False, return the cross-section per atom in cm^2.
        """
        res = self._coherent_scattering(energy)
        if not mass:
            res *= self.atomic_mass.rescale('g')
        return res

    def incoherent_scattering_cross_section(self, energy, mass=True):
        """
        Return the incoherent-scattering cross section in cm^2/g as a function
        of energy. The energy must be within the range 100 < E < 8e5 eV.

        Cross-sections at energies below 250 eV should not be considered
        reliable.

        If *mass* is True, return the cross-section per gram in cm^2/g.
        If *mass* is False, return the cross-section per atom in cm^2.
        """
        res = self._incoherent_scattering(energy)
        if not mass:
            res *= self.atomic_mass.rescale('g')
        return res
