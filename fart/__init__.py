# -*- coding: utf-8 -*-

"""Top-level package for fart."""

__author__ = """Nicolas Aunai"""
__email__ = 'nicolas.aunai@lpp.polytechnique.fr'
__version__ = '0.1.0'


from .solver_tree import Solver, complex_integral, quad_integral, plot_tree, integrand_func
__ALL__ = ["Solver", "complex_integral", "quad_integral", "plot_tree", "integrand_func"]
