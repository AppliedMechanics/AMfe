"""
Test for amfe-tools module
"""

from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from amfe.tools import invert_dictionary, invert_dictionary_with_iterables
from .tools import CustomDictAssertTest


class ToolsTest(TestCase):
    def setUp(self):
        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_invert_dictionary(self):
        dictionary = {'a': 1,
                      2: 'bjk',
                      5: 'bjk',
                      9: (1, 2)}

        dictionary_desired = {1: 'a',
                              'bjk': [2, 5],
                              (1, 2): 9}

        dictionary_actual = invert_dictionary(dictionary)

        self.custom_asserter.assert_dict_equal(dictionary_actual, dictionary_desired)

    def test_invert_dictionary_with_iterables(self):
        dictionary = {'a': np.array([1, 3, 8]),
                      7: np.array([5, 8]),
                      True: (5,),
                      ('bc', 'de', 5): [2, 4],
                      None: np.array([4]),
                      'ij': ('tuple', 1),
                      False: ('tuple',),
                      3: 'de',
                      5: 'df'}

        dictionary_desired = {1: ('a', 'ij'),
                              3: np.array(['a'], dtype=object),
                              5: np.array([7, True], dtype=object),
                              8: np.array(['a', 7], dtype=object),
                              2: [('bc', 'de', 5)],
                              4: [('bc', 'de', 5), None],
                              'tuple': ('ij', False),
                              'd': 5,
                              'e': 3,
                              'f': 5}

        dictionary_actual = invert_dictionary_with_iterables(dictionary)

        self.custom_asserter.assert_dict_equal(dictionary_actual, dictionary_desired)

        dictionary = {'a': [8],
                      7: pd.Series({'col1': 5, 'col2': 8}),
                      ('bc', 'de', 5): (2, 4),
                      'ij': ('tuple', 1)}
        self.assertRaises(ValueError, invert_dictionary_with_iterables, dictionary)
