# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase

import amfe


class TestPackageSpecs(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_version(self):
        version = amfe.__version__
        self.assertIsInstance(version, str)
        self.assertGreaterEqual(len(version), 1)

    def test_authors(self):
        authors = amfe.__author__
        self.assertIsNotNone(authors)
