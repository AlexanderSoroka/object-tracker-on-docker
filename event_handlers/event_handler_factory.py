"""
Implementation of event handler factory
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


import importlib


class EventHandlerFactory:
    """
    Dynamically load handler module and instantiate a class from it
    """
    @staticmethod
    def create(module_name, class_name):
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        instance = class_()
        return instance
