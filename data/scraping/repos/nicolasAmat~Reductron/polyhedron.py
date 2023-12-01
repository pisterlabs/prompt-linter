
"""
Polyhedron Module

Equations provided by the `reduce` tool from the TINA toolbox.
TINA toolbox: http://projects.laas.fr/tina/

This file is part of Reductron.

Reductron is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Reductron is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Reductron. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

__author__ = "Nicolas AMAT, LAAS-CNRS"
__contact__ = "namat@laas.fr"
__license__ = "GPLv3"
__version__ = "1.0"

import re
import sys
from typing import Optional


class Polyhedron:
    """ Reduction equations system.

    Attributes
    ----------
    places_initial : set of str
        A set of place identifiers from the initial Petri net.
    places_reduced : set of str
        A set of place identifiers from the reduced Petri net.
    additional_initial : set of str
        A set of additional variables from the initial Petri net.
    additional_reduced : set of str
        A set of additional variables from the reduced Petri net.
    additional_vars : set of Variable
        A set of additional variables (not places).
    equations : list of Equation
        A list of (in)equations.
    """

    def __init__(self, filename: str, places_initial: set[str], places_reduced: set[str], additional_initial: set[str], additional_reduced: set[str]) -> None:
        """ Initializer.

        Parameters
        ----------
        filename : str
            Path to reduction system (.net format).
        places_initial : set of str
            A set of place identifiers from the initial Petri net.
        places_reduced : set of str
            A set of place identifiers from the reduced Petri net.
        additional_initial : set of str
            A set of additional variables from the initial Petri net.
        additional_reduced : set of str
            A set of additional variables from the reduced Petri net.
        """
        self.places_initial: set[str] = set(places_initial)
        self.places_reduced: set[str] = set(places_reduced)

        self.additional_initial : set[str] = additional_initial
        self.additional_reduced : set[str] = additional_reduced

        self.additional_vars: set[str] = set()

        self.equations: list[Equation] = []

        self.parser(filename)

    def __str__(self) -> str:
        """ Equations to textual format.

        Returns
        -------
        str
            Debugging format.
        """
        return '\n'.join(map(str, self.equations))

    def smtlib(self, k1: Optional[int] = None, k2: Optional[int] = None, common: Optional[int] = None) -> str:
        """ Declare the additional variables and assert the equations.

        Parameters
        ----------
        k1 : int, optional
            Order for the initial places.
        k2 : int, optional
            Order for the reduced places.
        common : int, optional
            Order for common places.

        Returns
        --------
        str
            SMT-LIB format.
        """
        smt_input = ' '.join(map(lambda eq: eq.smtlib(k1, k2, common), self.equations))
        
        if len(self.equations) > 1:
            smt_input = "(and {})".format(smt_input)

        if self.additional_vars:
            smt_input = "(exists ({}) {})".format(''.join(map(lambda var: "({} Int)".format(var), self.additional_vars)), smt_input)

        return smt_input

    def smtlib_declare(self, k1: Optional[int] = None, k2: Optional[int] = None, common: Optional[int] = None, exclude_initial: bool = False, exclude_reduced: bool = False) -> list[str]:
        """ Declare variables.

        Parameters
        ----------
        k1 : int, optional
            Order for the initial places.
        k2 : int, optional
            Order for the reduced places.
        common : int, optional
            Order for common places.
        exclude_initial : bool, optional
            Exclude declaration for the initial places.
        exclude_reduced : bool, optional
            Exclude declaration for the reduced places.

        Returns
        --------
        list of str
            SMT-LIB format.
        """
        def place_smtlib(place, k):
            return place if k is None else "{}@{}".format(place, k)

        declaration = []

        if not exclude_initial or not exclude_reduced:
            for place in set(self.places_initial) & set(self.places_reduced):
                declaration.append(place_smtlib(place, common))

        if not exclude_initial:
            for place in set(self.places_initial) - set(self.places_reduced):
                declaration.append(place_smtlib(place, k1))

        if not exclude_reduced:
            for place in set(self.places_reduced) - set(self.places_initial):
                declaration.append(place_smtlib(place, k2))

        return declaration

    def parser(self, filename: str) -> None:
        """ System of reduction equations parser.
            
        Parameters
        ----------
        filename : str
            Path to reduction system (.net format).

        Raises
        ------
        FileNotFoundError
            Reduction system file not found.
        """
        try:
            with open(filename, 'r') as fp:
                content = re.search(r'generated equations\n(.*)?\n\n',
                                    fp.read().replace('#', '.').replace(',', '.'), re.DOTALL)  # '#' and ',' forbidden in SMT-LIB
                if content:
                    for line in re.split('\n+', content.group())[1:-1]:
                        if line.partition(' |- ')[0] not in ['. O', '. C']:
                            self.equations.append(
                                Equation(line, self))
            fp.close()
        except FileNotFoundError as e:
            sys.exit(e)


class Equation:
    """ Reduction equation.

    Attributes
    ----------
    left : list of Variable
        A left members (sum).
    right : list of Variable
        A right members (sum).
    operator : str
        An operator (=, <=, >=, <, >).
    """

    def __init__(self, content: str, polyhedron: Polyhedron) -> None:
        """ Initializer.

        Parameters
        ----------
        content : str
            Equation to parse.
        polyhedron : Polyhedron
            Current system of reduction equations.
        """
        self.left: list[Variable] = []
        self.right: list[Variable] = []
        self.operator: str = ""

        self.parse_equation(content, polyhedron)

    def __str__(self) -> str:
        """ Equation to .net format.

        Returns
        -------
        str
            Debugging format.
        """
        return ' + '.join(map(str, self.left)) + ' = ' + ' + '.join(map(str, self.right))

    def smtlib(self, k1: Optional[int] = None, k2: Optional[int] = None, common: Optional[int] = None) -> str:
        """ Assert the Equation.

        Parameters
        ----------
        k1 : int, optional
            Order for the initial places.
        k2 : int, optional
            Order for the reduced places.
        common : int, optional
            Order for common places.
        
        Returns
        -------
        str
            SMT-LIB format.
        """
        return "({} {} {})".format(self.operator, self.member_smtlib(self.left, k1=k1, k2=k2, common=common), self.member_smtlib(self.right, k1=k1, k2=k2, common=common))

    def member_smtlib(self, member: list[Variable], k1: Optional[int] = None, k2: Optional[int] = None, common: Optional[int] = None) -> str:
        """ Helper to assert a member (left or right).

        Parameters
        ----------
        member : list of Variable
            One of the two members (left or right).
        k1 : int, optional
            Order for the initial places.
        k2 : int, optional
            Order for the reduced places.
        common : int, optional
            Order for common places.
    
        Returns
        -------
        str
            SMT-LIB format.
        """
        smt_input = ' '.join(map(lambda var: var.smtlib(k1=k1, k2=k2, common=common), member))

        if len(member) > 1:
            smt_input = "(+ {})".format(smt_input)

        return smt_input

    def parse_equation(self, content: str, polyhedron: Polyhedron) -> None:
        """ Equation parser.

        Parameters
        ----------
        content : str
            Content to parse (.net format).
        polyhedron : Polyhedron
            Current polyhedron.
        """
        elements = re.split(r'\s+', content.partition(' |- ')[2])

        current, inversed = self.left, self.right
        minus = False

        for element in elements:

            if not element:
                continue

            if element in ['=', '<=', '>=', '<', '>']:
                self.operator = element
                current, inversed = inversed, current
                minus = False
                continue

            if element == '+':
                minus = False
                continue

            if element == '-':
                minus = True
                continue

            multiplier = None

            # `convert` specific case
            if '-1.' in element:
                element = element.replace('-1.', '')
                minus ^= True
                if minus:
                    current.append(Variable('0', multiplier))

            elif element.rfind('.') > element.rfind('}'):
                index = element.rfind('.')
                multiplier, element = element[:index], element[index+1:]

            variable = element.replace('{', '').replace('}', '')
            instantiated_variable = self.check_variable(variable, multiplier, polyhedron)

            if not minus:
                current.append(instantiated_variable)
            else:
                inversed.append(instantiated_variable)

    def check_variable(self, variable: str, multiplier: Optional[int], polyhedron: Polyhedron) -> None:
        """ Check from which is a variable, and instantiate it.

        Parameters
        ----------
        element : str
            Variable of integer constant.
        system : System
            Current system of reduction equations.

        Returns
        -------
        Variable
            Instantiated variable.
        """
        from_initial, from_reduced, from_additional, from_coherency_constraint = False, False, False, False

        if not variable.isnumeric():
            if variable in polyhedron.places_initial:
                from_initial = True
            if variable in polyhedron.places_reduced:
                from_reduced = True
            if variable in polyhedron.additional_initial or variable in polyhedron.additional_reduced:
                from_coherency_constraint = True
            if not (from_initial or from_reduced or from_coherency_constraint):
                from_additional = True
                polyhedron.additional_vars.add(variable)

        instantiated_variable = Variable(variable, multiplier, from_initial, from_reduced, from_coherency_constraint, from_additional)

        return instantiated_variable


class Variable:
    """ Variable.

    Note
    ----
    May be constant value. 

    Attributes
    ----------
    id : str
        An identifier.
    multiplier : str, optional
        A multiplier (if there is one).
    from_initial : bool
        Is from initial net.
    from_reduced : bool
        Is from reduced net.
    from_coherency_constraint : bool, optional
            Is from coherency constraint.
    from_additional : bool
        Is from additional variables.
    """

    def __init__(self, id: str, multiplier: Optional[str] = None, from_initial: bool = False, from_reduced: bool = False, from_coherency_constraint: bool = False, from_additional: bool = False) -> None:
        """ Initializer.

        Parameters
        ----------
        id : str
            An identifier.
        multiplier : int, optional
            A multiplier.
        from_initial : bool, optional
            Is from initial net.
        from_reduced : bool, optional
            Is from reduced net.
        from_coherency_constraint : bool, optional
            Is from coherency constraint.
        from_additional : bool, optional
            Is from additional variables.
        """
        self.id: str = id
        self.multiplier: Optional[str] = multiplier

        self.from_initial: bool = from_initial
        self.from_reduced: bool = from_reduced

        self.from_coherency_constraint: bool = from_coherency_constraint

        self.from_additional: bool = from_additional

    def __str__(self) -> str:
        """ Variable to textual format.

        Returns
        -------
        str
            Debugging format.
        """
        text = ""

        if self.multiplier is not None:
            text += "{}.".format(self.multiplier)

        return text + self.id

    def smtlib(self, k1: Optional[int] = None, k2: Optional[int] = None, common: Optional[int] = None) -> str:
        """ Assert the Variable and its multiplier if needed.

        Parameters
        ----------
        k1 : int, optional
            Order for the initial places.
        k2 : int, optional
            Order for the reduced places.
        common : int, optional
            Order for common places.

        Returns
        -------
        str
            SMT-LIB format.
        """
        smt_input = self.id

        if self.from_initial and self.from_reduced and common is not None:
            smt_input += "@{}".format(common)
        elif self.from_initial and k1 is not None:
            smt_input += "@{}".format(k1)
        elif self.from_reduced and k2 is not None:
            smt_input += "@{}".format(k2)

        if self.multiplier is not None:
            smt_input = "(* {} {})".format(self.multiplier, smt_input)

        return smt_input

