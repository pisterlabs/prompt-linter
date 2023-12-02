# encoding: utf-8

from collections import defaultdict, namedtuple
from itertools import product, chain
from toposort import toposort
import logging

from .shared import PRIOR, POSTERIOR, NOREL, EQUAL, INIT, OL_PARENT, UNCL, LAC
from .pre_genealogical_coherence import Coherence
logger = logging.getLogger(__name__)


class ParentCombination(object):
    def __init__(self, parent, rank, perc, gen, prior=None, posterior=None, undirected=False):
        self.parent = parent  # witness label
        self.rank = rank  # integer rank
        self.perc = perc  # percentage coherence
        self.gen = gen  # generation (e.g. 1=parent, 2=grandparent, ...)
        self.prior = prior  # integer number of prior readings in the parent - or None=undefined
        self.posterior = posterior  # integer number of posterior readings in the parent - or None=undefined
        self.undirected = undirected  # boolean - is this relationship undirected?

    @property
    def strength(self):
        if self.prior is None or self.posterior is None:
            return -1

        return self.prior - self.posterior

    def __repr__(self):
        return ("<Parent Combination: parent={}, rank={}, perc={}, gen={}, prior={}, posterior={}, "
                "strength={}, undirected={}>".format(
            self.parent, self.rank, self.perc, self.gen, self.prior, self.posterior, self.strength, self.undirected))

    def __eq__(self, other):
        """
        Provide a simple test of equality, base on our input data
        """
        this = (self.parent, self.rank, self.perc, self.gen, self.prior, self.posterior, self.undirected)
        that = (other.parent, other.rank, other.perc, other.gen, other.prior, other.posterior, other.undirected)
        return this ==  that

    def __hash__(self):
        """
        Generate a hash on the same principle that the __eq__method uses to test equality
        """
        return hash((self.parent, self.rank, self.perc, self.gen, self.prior, self.posterior, self.undirected))

class TooManyAborts(Exception):
    pass


class CyclicDependency(Exception):
    pass


class ReadingRelationship(object):
    """
    Class representing a reading in a specified variant unit.
    """
    def __init__(self, variant_unit, reading, cursor):
        self.variant_unit = variant_unit
        self.reading = reading
        self.cursor = cursor
        self._recursion_history = []
        self._abort_count = 0

    def identify_relationship(self, other_reading):
        """
        Find out how our reading is related to this other one

        Returns EQUAL, PRIOR, POSTERIOR, UNCL or NOREL
        """
        if self.reading == other_reading:
            return EQUAL

        # Even though some readings have multiple parents (c&d), the question
        # here is not 'does X explain Y completely?' but instead it's 'which of
        # X and Y is PRIOR?' Local stemmata are not allowed loops, so we can
        # always answer that question.

        def check(reading, desired_parent):
            bits = [x.strip() for x in desired_parent.split('&')]  # len 1 or more
            for bit in bits:
                if reading == bit:
                    # We matched one required parent reading
                    return True
            return False

        r2_ancestor = self.get_parent_reading(other_reading)
        if check(self.reading, r2_ancestor):
            return PRIOR

        r1_ancestor = self.get_parent_reading(self.reading)
        if check(other_reading, r1_ancestor):
            return POSTERIOR

        if UNCL == r1_ancestor or UNCL == r2_ancestor:
            return UNCL

        return NOREL

    def get_parent_reading(self, reading):
        """
        Get the parent reading for this reading
        """
        sql = """SELECT parent FROM cbgm
                 WHERE variant_unit = ?
                 AND label = ?"""
        self.cursor.execute(sql, (self.variant_unit, reading))

        row = self.cursor.fetchone()
        if row is None:
            logger.warning("No parent reading found for %s reading %s - returning UNCL", self.variant_unit, reading)
            return UNCL
        else:
            return row[0]


class GenealogicalCoherence(Coherence):
    """
    Class representing genealogical coherence (potential ancestors)
    """
    def __init__(self, *o, min_strength=None, **k):
        super().__init__(*o, **k)

        self.columns.insert(2, 'D')
        self.columns.extend(["W1<W2",  # Prior variants in W2
                             "W1>W2",  # Posterior variants in W2
                             "UNCL",
                             "NOREL"])

        # Dict of witness-reading relationships
        # {W2: {variant_unit: relationship, }, }
        self.reading_relationships = defaultdict(dict)
        self._parent_search = set()
        self._done_cycle_check = False
        self.min_strength = min_strength

        if self.min_strength:
            # The normal cached coherence values will be wrong if we want min strength...
            self._cache_key += '.min_strength.{}'.format(self.min_strength)

    def _detect_cycles(self):
        """
        Search for cycles in our data
        """
        if self._done_cycle_check or self.variant_unit is None:
            return

        # Check for bad data
        data = defaultdict(set)
        sql = """SELECT label, parent FROM cbgm
                 WHERE variant_unit = ?
                 """
        self.cursor.execute(sql, (self.variant_unit,))
        for row in self.cursor:
            data[row[0]].add(row[1])
        try:
            list(toposort(data))
        except ValueError:
            # There's a cycle in our data...
            raise CyclicDependency

        self._done_cycle_check = True

    def generate(self):
        """
        Sub-classed method that hides rows that aren't potential ancestors
        """
        # We might not have had a variant unit when we generated, so we need
        # to offer to detect cycles every time.
        self._detect_cycles()

        if self._already_generated:
            return

        if self.use_cache and self._check_cache():
            self._load_cache()
            return

        logger.debug("Generating genealogical coherence data for %s", self.w1)

        self._calculate_reading_relationships()

        self._generate_rows()

        new_rows = []
        for row in self.rows:
            if row['W1>W2'] > row['W1<W2']:
                # W1 has more prior variants than W2 - so W2 isn't a
                # potential ancestor
                continue

            new_rows.append(row)

        self.rows = new_rows

        # Now re-sort
        self._sort()

        self._already_generated = True
        logger.debug("Generated genealogical coherence data for %s", self.w1)

        if self.use_cache:
            self._store_cache()

    def _calculate_reading_relationships(self):
        """
        Populates the self.reading_relationships dictionary.

        Possible relationships are:
            PRIOR (self.w1's reading is directly prior to w2's)
            POSTERIOR (self.w1's reading is directly posterior to w2's)
            UNCL (one or other of w1 and w2 has an unclear parent)
            NOREL (no direct relationship between the readings)
            EQUAL (they're the same reading)
        """
        # Find every variant unit in which we're extant
        sql = "SELECT variant_unit, label FROM cbgm WHERE witness = ?"
        for vu, label in list(self.cursor.execute(sql, (self.w1,))):
            reading_obj = ReadingRelationship(vu, label, self.cursor)
            for w2 in self.all_mss:
                if w2 == self.w1:
                    continue
                attestation = self.get_attestation(w2, vu)
                if attestation is None:
                    # Nothing for this witness at this place
                    continue
                w2_label = attestation
                if w2_label == LAC:
                    # lacuna
                    continue
                rel = reading_obj.identify_relationship(w2_label)
                self.reading_relationships[w2][vu] = rel

    def _add_D(self, w2, row):
        """
        Direction - this is used in the same way as the CBGM's genealogical
        queries program. So, it shows '-' for no direction.

        Additionally, I use it to show weak textual flow, if self.min_strength
        has been set.
        """
        if 'W1<W2' not in row:
            return False
        if 'W1>W2' not in row:
            return False

        if row['W1<W2'] == row['W1>W2']:
            row['D'] = '-'  # no direction
            row['NR'] = 0  # so rank 0
        elif self.min_strength and (row['W1<W2'] - row['W1>W2']) < self.min_strength:
            # We will make these act like non-direction relationships
            row['D'] = 'w'  # too weak
            row['NR'] = 0  # so rank 0
        else:
            row['D'] = ''

        return True

    def _add_W1_lt_W2(self, w2, row):
        """
        How many times W2 has prior variants to W1
        """
        row['W1<W2'] = len([x for x in list(self.reading_relationships[w2].values())
                            if x == POSTERIOR])
        return True

    def _add_W1_gt_W2(self, w2, row):
        """
        How many times W2 has posterior variants to W1
        """
        row['W1>W2'] = len([x for x in list(self.reading_relationships[w2].values())
                            if x == PRIOR])
        return True

    def _add_UNCL(self, w2, row):
        """
        Count how many passages are unclear
        """
        uncls = [k for k, v in self.reading_relationships[w2].items()
                 if v == UNCL]
        if uncls and self.debug:
            print("UNCL with {} in {}".format(w2, ', '.join(uncls)))
        row['UNCL'] = len(uncls)

        return True

    def _add_NOREL(self, w2, row):
        """
        Count in how many passages W2's reading has no relation to W1's reading
        """
        if 'W1<W2' not in row:
            return False
        if 'W1>W2' not in row:
            return False
        if 'UNCL' not in row:
            return False
        if 'PASS' not in row:
            return False
        if 'EQ' not in row:
            return False
        row['NOREL'] = (row['PASS'] -
                        row['EQ'] -
                        row['UNCL'] -
                        row['W1>W2'] -
                        row['W1<W2'])

        # Double check all the logic:
        norel_p = [x for x, y in list(self.reading_relationships[w2].items())
                   if y == NOREL]
        assert row['NOREL'] == len(norel_p), (
            w2,
            row['NOREL'],
            row['PASS'],
            row['EQ'],
            row['UNCL'],
            row['W1>W2'],
            row['W1<W2'],
            self.reading_relationships[w2],
            len(self.reading_relationships[w2]),
            norel_p)
        if norel_p and self.debug:
            print("NOREL with {} in {}".format(w2, ', '.join(norel_p)))

        return True

    def potential_ancestors(self):
        """
        Return a list of potential ancestors. This respects the work done in self.add_D above.
        """
        self.generate()
        return [x['W2'] for x in self.rows
                if x['NR'] != 0]

    def parent_combinations(self, reading, parent_reading, *, max_rank=None, min_perc=None, include_undirected=False,
                            my_gen=1):
        """
        Return a list of possible parent combinations that explain this reading.

        If the parent_reading is of length 3 (e.g. c&d&e) then the combinations
        will be length 3 or less.

        Returns a list of lists or ParentCombination objects, e.g.:
            [
             # 05 explains this reading by itself
             [('05' = witness, 4 = rank, 1 = generation)],

             # 03 and P75 are both required to explain this reading and both
             # are generation 2 (e.g. attest a parent reading)
             [('03', 3, 2), ('P75', 6, 2)],

             # A explains this reading by itself but it is generation 3 - in
             # other words all witnesses attesting our parent readings all
             # have A as their parent (one with rank 6 and one with rank 4)
             [('A', 6, 3), ('A', 4, 3)],
             ...
             ]
        """
        assert self.variant_unit, "You must set a variant unit before calling parent_combinations"

        logger.debug("parent_combinations: vu=%s, reading=%s, parent=%s, max_rank=%s, min_perc=%s, my_gen=%s",
                     self.variant_unit, reading, parent_reading, max_rank, min_perc, my_gen)

        assert not (max_rank and min_perc), "You can't specify both max_rank and min_perc"

        self.generate()
        if my_gen == 1:
            # top level
            self._parent_search = set()

        ret = []
        potanc = self.potential_ancestors()
        # Things that explain it by themselves:
        for row in self.rows:
            undirected = False
            # Check the real rank (_NR) - so joint 6th => 6. _RANK here could be
            # 7, 8, 9 etc. for joint 6th.
            if max_rank is not None and row['_NR'] > max_rank:
                # Exceeds connectivity setting
                continue

            if min_perc is not None and row['PERC1'] < min_perc:
                # Coherence percentage is too low
                continue

            if row['W2'] not in potanc:
                if include_undirected:
                    undirected = True
                else:
                    # Not a potential ancestor (undirected genealogical coherence or too weak)
                    logger.debug("Ignoring %s as it's not a potential ancestor", row)
                    continue

            if row['READING'] == reading:
                # This matches our reading and is within the connectivity threshold - take it

                # This is in a row for W2, but we want the prior readings in the PARENT.
                # So We need the W1<W2 entry (which is the posterior readings in the child.)
                # And vice versa for the posterior count.
                prior = row['W1<W2']
                posterior = row['W1>W2']
                if self.min_strength and not include_undirected:
                    assert prior - posterior >= self.min_strength, "This row shouldn't be a potential ancestor: {}".format(row)

                ret.append([ParentCombination(row['W2'], row['_NR'], row['PERC1'],
                                              my_gen, prior, posterior, undirected)])

        if parent_reading in (INIT, OL_PARENT, UNCL):
            # No parents - nothing further to do
            return ret

        # Now the parent reading
        partial_explanations = []
        bits = [x.strip() for x in parent_reading.split('&')]
        if len(bits) == 1:
            next_gen = my_gen + 1
        else:
            next_gen = my_gen

        for partial_parent in bits:
            if partial_parent in self._parent_search:
                # Already been here - must be looping...
                continue

            self._parent_search.add(partial_parent)

            if partial_parent == INIT:
                # Simple - who reads INIT?
                partial_explanations.append(
                    self.parent_combinations(INIT, None, max_rank=max_rank, min_perc=min_perc,
                                             include_undirected=include_undirected, my_gen=my_gen + 1))
                continue
            if partial_parent == OL_PARENT:
                # Simple - who reads OL_PARENT?
                partial_explanations.append(
                    self.parent_combinations(OL_PARENT, None, max_rank=max_rank, min_perc=min_perc,
                                             include_undirected=include_undirected, my_gen=my_gen + 1))
                continue

            # We need to recurse, and find out what combinations explain our
            # (partial) parent.
            reading_obj = ReadingRelationship(self.variant_unit,
                                              partial_parent,
                                              self.cursor)

            next_reading = partial_parent
            next_parent = reading_obj.get_parent_reading(partial_parent)

            if next_reading == reading and next_parent == parent_reading:
                # No point recursing just warn the user...
                logger.warning("Would recurse infinitely... w1=%s, vu=%s, reading=%s, parent=%s, partial_parent=%s",
                               self.w1, self.variant_unit, reading, parent_reading, partial_parent)
            else:
                expl = self.parent_combinations(next_reading, next_parent, max_rank=max_rank, min_perc=min_perc,
                                                include_undirected=include_undirected, my_gen=next_gen)
                partial_explanations.append(expl)

        if not partial_explanations:
            # We couldn't find anything
            return []

        if len(partial_explanations) == 1:
            # We've got a single parent - simple
            ret.extend(partial_explanations[0])
            return ret

        else:
            # We now combine the lists in such a way as to get the same structure
            # as above but now with (potentially) multiple tuples in the inner lists.
            prod = product(*partial_explanations)
            combined = list(list(set(chain(*x))) for x in prod)
            return combined


def generate_genealogical_coherence_cache(w1, db_file, min_strength=None):
    """
    Generate genealogical coherence (variant unit independent)
    and store a cached copy.
    """
    coh = GenealogicalCoherence(db_file, w1, pretty_p=False, use_cache=True, min_strength=min_strength)
    coh.generate()

    # A return of None is interpreted as abort, so just return True
    return True


def gen_coherence(db_file, w1, variant_unit=None, *, pretty_p=False, debug=False, use_cache=False, min_strength=None):
    """
    Show a table of potential ancestors of w1.

    If variant_unit is supplied, then two extra columns are output
    showing the reading supported by each witness.
    """
    coh = GenealogicalCoherence(db_file, w1, pretty_p=pretty_p, debug=debug, use_cache=use_cache,
                                min_strength=min_strength)
    if variant_unit:
        coh.set_variant_unit(variant_unit)
    title = "Potential ancestors for W1={}".format(w1)
    if min_strength:
        title += ' [min_strength={}]'.format(min_strength)
    return "{}\n{}".format(title, coh.tab_delim_table())
