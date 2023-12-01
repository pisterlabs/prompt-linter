from unittest import TestCase
import shutil
import logging
import tempfile
from CBGM.pre_genealogical_coherence import Coherence
from CBGM import test_db
from CBGM.test_logging import default_logging

default_logging()
logger = logging.getLogger(__name__)

TEST_DATA = """# -*- coding: utf-8 -*-
# This is a made up data set, purely for testing.

from CBGM.populate_db import Reading, LacunaReading, AllBut
from CBGM.shared import UNCL, INIT, OL_PARENT

all_mss = set(['B', 'C', 'D', 'E'])

struct = {
    '21': {
        '2': [
            Reading('a', 'ηθελον', AllBut('B'), INIT),
            Reading('b', 'ηλθον', ['B'], 'a')],
        '6-8': [
            Reading('a', 'λαβειν αυτον', AllBut('C', 'D'), UNCL),
            Reading('b', 'αυτον λαβειν', ['C'], UNCL),
            LacunaReading(['D'])],
    },
    '22': {
        '3': [
            Reading('a', '', AllBut('C'), INIT),
            Reading('b', 'τε', ['C'], 'a')],
        '20': [
            Reading('a', 'ιδων', ['B'], UNCL),
            Reading('b', 'ειδον', ['C'], UNCL),
            Reading('c', 'ειδεν', ['D'], UNCL),
            Reading('d', 'ειδως', ['E'], UNCL)],
    },
    '23': {
        '1': [
            Reading('a', '', AllBut('C', 'B'), INIT),
            Reading('b', 'και', ['B'], 'a'),
            LacunaReading(['C'])],
        '4-10': [
            Reading('a', 'ηλθεν πλοιαρια εκ τιβεριαδος', ['B'], UNCL),
            Reading('b', 'ηλθεν πλοια εκ τιβεριαδος', ['C'], UNCL),
            Reading('c', 'ηλθεν πλοια εκ της τιβεριαδος', ['D', 'E'], UNCL)],
    }
}
"""


B_ROWS = [{'W2': 'E', 'NR': 1, 'EQ': 2, 'PASS': 6, 'PERC1': 33.333333333333336, '_NR': 1, '_RANK': 1},
          {'W2': 'A', 'NR': '', 'EQ': 1, 'PASS': 3, 'PERC1': 33.333333333333336, '_NR': 1, '_RANK': 2},
          {'W2': 'D', 'NR': 3, 'EQ': 1, 'PASS': 5, 'PERC1': 20.0, '_NR': 3, '_RANK': 3},
          {'W2': 'C', 'NR': 4, 'EQ': 0, 'PASS': 5, 'PERC1': 0.0, '_NR': 4, '_RANK': 4}]


B_ROWS_AT_23_4_10 = [{'W2': 'E', 'NR': 1, 'EQ': 2, 'PASS': 6, 'PERC1': 33.333333333333336, '_NR': 1,
                      '_RANK': 1, 'READING': 'c', 'TEXT': 'ηλθεν πλοια εκ της τιβεριαδος'},
                     {'W2': 'A', 'NR': '', 'EQ': 1, 'PASS': 3, 'PERC1': 33.333333333333336, '_NR': 1,
                      '_RANK': 2, 'READING': None, 'TEXT': None},
                     {'W2': 'D', 'NR': 3, 'EQ': 1, 'PASS': 5, 'PERC1': 20.0, '_NR': 3,
                      '_RANK': 3, 'READING': 'c', 'TEXT': 'ηλθεν πλοια εκ της τιβεριαδος'},
                     {'W2': 'C', 'NR': 4, 'EQ': 0, 'PASS': 5, 'PERC1': 0.0, '_NR': 4,
                      '_RANK': 4, 'READING': 'b', 'TEXT': 'ηλθεν πλοια εκ τιβεριαδος'}]

C_TABLE = """  W2    	   NR    	  PERC1  	   EQ    	  PASS  
   A    	    1    	 50.000 	    1    	    2   
   D    	    2    	 25.000 	    1    	    4   
   E    	    3    	 20.000 	    1    	    5   
   B    	    4    	 0.000 	         	    5   """


class TestCoherence(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = test_db.TestDatabase(TEST_DATA)
        cls.tmpdir = tempfile.mkdtemp(__name__)
        Coherence.CACHE_BASEDIR = cls.tmpdir

    @classmethod
    def tearDownClass(cls):
        cls.test_db.cleanup()
        shutil.rmtree(cls.tmpdir)

    def test_cache(self):
        """
        Check that the caching works
        """
        # Populate the empty cache
        coh1 = Coherence(self.test_db.db_file, 'B', pretty_p=False, use_cache=True)
        self.assertFalse(coh1._check_cache())
        coh1.generate()
        self.assertEqual(coh1.rows, B_ROWS)

        # Retrieve from cache
        coh2 = Coherence(self.test_db.db_file, 'B', pretty_p=False, use_cache=True)
        self.assertTrue(coh2._check_cache())
        coh2._load_cache()
        self.assertEqual(coh2.rows, B_ROWS)

        # Shouldn't be in cache (pretty_p different)
        coh3 = Coherence(self.test_db.db_file, 'B', pretty_p=True, use_cache=True)
        self.assertFalse(coh3._check_cache())

        # Shouldn't be in cache (w1 different)
        coh4 = Coherence(self.test_db.db_file, 'C', pretty_p=False, use_cache=True)
        self.assertFalse(coh4._check_cache())

    def test_generate(self):
        """
        Test that the generate function produces rows, and they
        contain the right data for w1=B.
        """
        coh = Coherence(self.test_db.db_file, 'B')
        self.assertFalse(coh.rows)
        coh.generate()
        self.assertEqual(B_ROWS, coh.rows)

    def test_set_variant_unit(self):
        coh = Coherence(self.test_db.db_file, 'B')
        coh.generate()
        self.assertEqual(B_ROWS, coh.rows)
        coh.set_variant_unit('23/4-10')
        self.assertEqual(B_ROWS_AT_23_4_10, coh.rows)

    def test_all_attestations(self):
        coh = Coherence(self.test_db.db_file, 'B')
        att = coh.all_attestations()
        self.assertTrue(att['E']['21/2'], 'a')
        self.assertTrue(att['E']['22/20'], 'd')
        self.assertTrue(att['C']['21/6-8'], 'b')
        self.assertTrue(att['B']['23/1'], 'b')
        self.assertTrue(att['A']['22/3'], 'a')
        self.assertNotIn('23/4-10', att['A'])

    def test_get_attestation(self):
        coh = Coherence(self.test_db.db_file, 'D')
        att = coh.get_attestation('C', '21/2')
        self.assertTrue(att, 'a')

    def test_tab_delim_table(self):
        coh = Coherence(self.test_db.db_file, 'C')
        tab = coh.tab_delim_table()
        self.assertEqual(tab, C_TABLE)
