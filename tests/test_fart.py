#!/usr/bin/env python

# -*- coding: utf-8 -*-


"""Tests for `whampyr` package."""

import pytest
from fart import Solver, complex_integral, quad_integral, plot_tree
import numpy as np



""" First test : the Nth root of a complex"""
def f1(z, k=0):
    a = 1/2 + 1j*np.sqrt(3)/2
    return z**11 - a

def df1(z, k=0):
    """Dreivative of f1"""
    return 11*z**10


@pytest.mark.test_id(1501)
def test_Number_of_zeros():
    """Lets just calculate the numbers of Zeros"""
    k = None
    domain = (-2.0, -2.0, 2.0, 2.0)

    def integrand_func(z, k, n=0):
        ret = -1j * 0.5 / np.pi * z ** n * df1(z) / f1(z)
        return ret

    v = complex_integral(integrand_func, domain, k, integr=quad_integral, n=0)

    assert int(np.round(np.real(v))) == 11


def f3(z, k=0):
    return z**50 + z**12 - 5*np.sin(20*z)*np.cos(12*z) - 1

def df3(z, k=0):
    """Dreivative of f2"""
    v = 50*z**49 + 12*z**11 - 5*(20*np.cos(20*z)*np.cos(12*z) -
                                  12*np.sin(12*z)*np.sin(20*z))
    return v

def test_Manyzeros():
    """Test the f3 function"""
    k = None
    domain = (-20.3, -5.0, 20.7, 5.1)
    def integrand_func(z, k, n=0):
        ret = -1j * 0.5 / np.pi * z ** n * df3(z) / f3(z)
        return ret

    v = complex_integral(integrand_func, domain, k, integr=quad_integral, n=0)

    assert int(np.round(np.real(v))) == 424


def test_calculate_zeros():
    """Try to calculate the zeros of f1"""

    k = None
    domain = (-2.0, -2.0, 2.0, 2.0)

    s = Solver(f1, domain, k,
               max_depth=20, deriv_function=df1)

    s.solve()
    s.clean_double_roots()

    print(len(s.zeros))

    assert  len(s.zeros) == 11



def test_calculate_zeros_Many():
    """Try to calculate the zeros of f3"""

    k = 1
    domain = (-20.3, -5.0, 20.7, 5.1)

    s = Solver(f3, domain, k,
               max_depth=20, deriv_function=df3)

    s.solve()
   # s.clean_double_roots()

    #plot_tree(s, f3, domain, 2)

    print(len(s.zeros))

    assert 424 == len(s.zeros)
