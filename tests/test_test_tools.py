"""
Test for tools module of test-tools
"""

from unittest import TestCase, expectedFailure
import numpy as np
from .tools import CustomDictAssertTest
from copy import copy


class CustomAssertMethodsTest(TestCase):
    def setUp(self):
        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_assert_dict_almost_equal(self):
        dict1 = {'key1': np.array([0.00000000000000000000001, 5.400000000000000000000000001]),
                  'key2': 5,
                  (1, 2): 'value'}
        dict2 = {'key1': np.array([0.0, 5.4]),
                  'key2': 5,
                  (1, 2): 'value'}

        self.custom_asserter.assert_dict_almost_equal(dict1, dict2)

        dict1 = {'key1': {'key2': {'key3': {'key4': {'key4': {'key5': 5}}}}}}
        dict2 = copy(dict1)
        self.custom_asserter.assert_dict_almost_equal(dict1, dict2)

        dict1 = {'key1': {'key2': {'key3': {'key4': {'key4': {'key5': 5}}}}}}
        dict2 = copy(dict1)
        self.custom_asserter.assert_dict_almost_equal(dict1, dict2)

    @expectedFailure
    def test_assert_dict_almost_equal_failure(self):
        dict1 = {'key1': np.array([0.00001, 5.40001]),
                 'key2': 5,
                 (1, 2): 'value'}
        dict2 = {'key1': np.array([0.0, 5.4]),
                 'key2': 5,
                 (1, 2): 'value'}

        self.custom_asserter.assert_dict_almost_equal(dict1, dict2)

    @expectedFailure
    def test_assert_dict_almost_equal_limit_failure(self):
        dict1 = {'key1': {
            'key2': {'key3': {'key4': {'key4': {'key5': {'key6': {'key7': {'key8': {'key9': {'key10': 10}}}}}}}}}}}
        dict2 = copy(dict1)
        self.custom_asserter.assert_dict_almost_equal(dict1, dict2)

    def test_assert_dict_equal(self):
        dict1 = {'key1': np.array([0, 5]),
                  'key2': 5,
                  (1, 2): 'value'}
        dict2 = {'key1': np.array([0, 5]),
                  'key2': 5,
                  (1, 2): 'value'}

        self.custom_asserter.assert_dict_equal(dict1, dict2)

    @expectedFailure
    def test_assert_dict_equal_failure(self):
        dict1 = {'key1': np.array([0.00000000000000000000001, 5.400000000000000000000000001]),
                 'key2': 5,
                 (1, 2): 'value'}
        dict2 = {'key1': np.array([0.0, 5.4]),
                 'key2': 5,
                 (1, 2): 'value'}

        self.custom_asserter.assert_dict_equal(dict1, dict2)

    @expectedFailure
    def test_assert_dict_equal_limit_failure(self):
        dict1 = {'key1': {
            'key2': {'key3': {'key4': {'key4': {'key5': {'key6': {'key7': {'key8': {'key9': {'key10': 10}}}}}}}}}}}
        dict2 = copy(dict1)
        self.custom_asserter.assert_dict_almost_equal(dict1, dict2)

