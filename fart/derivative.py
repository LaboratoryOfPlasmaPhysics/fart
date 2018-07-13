""" the interpolation of the derivative"""

import numpy as np
from scipy.interpolate import interp1d


#  def num_prim_np(func,w_tab, *args):
def num_prim_np(vals, dw, *args):
    """simply compute the numerical derivative"""
    return np.gradient(vals, dw)


class interp_prim():
    """A class for the interpolation"""

    def __init__(self, func, w_tab,  **func_kwargs):

        dw = w_tab[1] - w_tab[0]

        if np.abs(dw.real) > np.abs(dw.imag):
            which_axis = 'real'
        else:
            which_axis = 'imag'

        self.which_axis = which_axis
        # print("Evoluting in the ", which_axis, "axis")

        if which_axis == "real":
            xs = w_tab.real
            res = w_tab[0].imag * 1.j
        else:
            xs = w_tab.imag
            res = w_tab[0].real

        function_values = func( w_tab, **func_kwargs)
        deriv_values = num_prim_np(function_values, w_tab[1] - w_tab[0])

        self.interp_prim_real = interp1d(xs, deriv_values.real, bounds_error=False, fill_value="extrapolate",
                                         )
        self.interp_prim_imag = interp1d(xs, deriv_values.imag, bounds_error=False, fill_value="extrapolate",
                                         )

    def compute_interp(self, w):
        if self.which_axis == "real":
            x = w.real
        else:
            x = w.imag

        return self.interp_prim_real(x) + self.interp_prim_imag(x) * 1.j


def complex_integral_num(f, rect, integr, n=0, **func_kwargs):
    """Contour integral on a box of size L
    we compute f' via the numerical derivative of f"""
    x0, y0, x1, y1 = rect
    Lx, Ly = x1 - x0, y1 - y0
    epsilon = 0.001

    def real_integrand_interpolated(f, N, **func_kwargs):
        x = np.linspace(x0 - epsilon * Lx, x1 + epsilon * Lx, N)

        w_r1 = x + 1j * y0
        f_prim_1 = interp_prim(f, w_r1, **func_kwargs)

        w_r2 = x + 1j * y1
        f_prim_2 = interp_prim(f, w_r2, **func_kwargs)

        g1 = f_prim_1.compute_interp
        g2 = f_prim_2.compute_interp
        return g1, g2

    def imag_integrand_interpolated(f, N, **func_kwargs):
        x = np.linspace(y0 - epsilon * Ly, y1 - epsilon * Lx, N)

        w_r1 = x * 1.j + x0
        f_prim_1 = interp_prim(f, w_r1, **func_kwargs)

        w_r2 = x * 1.j + x1
        f_prim_2 = interp_prim(f, w_r2, **func_kwargs)

        g1 = f_prim_1.compute_interp
        g2 = f_prim_2.compute_interp
        return g2, g1

    def real_integrand(x, n, **func_kwargs):
        w_r1 = x + 1j * y0
        w_r2 = x + 1j * y1

        g1 = w_r1 ** n * df1(w_r1) / f(w_r1, **func_kwargs)
        g2 = w_r2 ** n * df3(w_r2) / f(w_r2, **func_kwargs)
        return g1 - g2

    def imaginary_integrand(y, n, **func_kwargs):
        w_r1 = x1 + 1j * y
        w_r2 = x0 + 1j * y

        g1 = w_r1 ** n * df2(w_r1) / f(w_r1, **func_kwargs)
        g2 = w_r2 ** n * df4(w_r2) / f(w_r2, **func_kwargs)
        return g1 - g2

    [df1, df3], [df2, df4] = (real_integrand_interpolated(f, N=400, **func_kwargs),
                              imag_integrand_interpolated(f, N=400, **func_kwargs))

    integral1 = integr(real_integrand, x0, x1, n, **func_kwargs)
    integral2 = integr(imaginary_integrand, y0, y1, n, **func_kwargs)

    # val = r1 - I2 + j ( I1 + R2) = 1 + j 2
    val = integral1[0] + 1j * integral2[0]

    return val / (2.j * np.pi)


def complex_integral_num_raw(f, rect, n=0, **func_kwargs):
    """Contour integral on a box of size L
    we compute f' via the numerical derivative of f"""
    x0, y0, x1, y1 = rect
    Lx, Ly = x1 - x0, y1 - y0
    epsilon = 0.01

    def real_integrand(x, n, **func_kwargs):
        w_r1 = x + 1j * y0
        w_r2 = x + 1j * y1
        dw = x[1] - x[0]
        func_values1 = f(w_r1, **func_kwargs)
        dfunc_values1 = num_prim_np(func_values1, dw)

        func_values2 = f(w_r2, **func_kwargs)
        dfunc_values2 = num_prim_np(func_values2, dw)

        g1 = w_r1 ** n * dfunc_values1 / func_values1
        g2 = w_r2 ** n * dfunc_values2  / func_values2
        return g1 - g2

    def imaginary_integrand(y, n, **func_kwargs):
        w_r1 = x1 + 1j * y
        w_r2 = x0 + 1j * y

        dw = 1j*( y[1] - y[0])
        func_values1 = f(w_r1, **func_kwargs)
        dfunc_values1 = num_prim_np(func_values1, dw)

        func_values2 = f(w_r2, **func_kwargs)
        dfunc_values2 = num_prim_np(func_values2, dw)

        g1 = w_r1 ** n * dfunc_values1 / func_values1
        g2 = w_r2 ** n * dfunc_values2 / func_values2

        return g1 - g2

    Nlinspace=500
    xs = np.linspace(x0, x1, Nlinspace)

    vals = real_integrand(xs, n, **func_kwargs)

    integral1 = np.sum(vals[1:] + vals[:-1])*(x1-x0)/Nlinspace/2
    #integral1 = np.sum(vals)*(x1-x0)/Nlinspace

    xs = np.linspace(y0, y1, Nlinspace)

    vals = imaginary_integrand(xs, n, **func_kwargs)
    #integral2 = np.sum(vals)*(x1-x0)/Nlinspace

    integral2 = np.sum(vals[1:] + vals[:-1])*(x1-x0)/Nlinspace/2


    # val = r1 - I2 + j ( I1 + R2) = 1 + j 2
    val = integral1 + 1j * integral2

    return val / (2.j * np.pi)


def raw_integral(g, x0, x1, n=0, N = 90, **func_kwargs):
    """ calculate a raw integral of the complex function
    Should be faster than scipy.quad"""
    xs = np.linspace(x0,x1,N)
    value = np.sum(g(xs,n, **func_kwargs))
    return [value*(x1-x0)/N]
