#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Abstract class of assembly algorithms.
"""

__all__ = [
    'Assembly'
]


class Assembly:
    """
    Super class for all assemblies providing observer utilities.
    """

    def __init__(self):
        self._observers = list()
        return

    def add_observer(self, observer):
        self._observers.append(observer)
        return

    def remove_observer(self, observer):
        self._observers.remove(observer)
        return

    def notify(self):
        for observer in self._observers:
            observer.update(self)
        return

    def update(self, obj):
        pass
