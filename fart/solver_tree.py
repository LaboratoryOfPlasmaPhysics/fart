

from .tree import Node, QuadTree
import numpy as np
from scipy.integrate import quad
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from .derivative import complex_integral_num, complex_integral_num_raw


def raw_integral(g, x0, x1, k, n=0, N = 90):
    """ calculate a raw integral of the complex function
    Should be faster than scipy.quad"""
    xs = np.linspace(x0,x1,N)
    value = np.sum(g(xs, k,n))
    return [value*(x1-x0)/N]


def quad_integral(g, a, b, k,n=0):
    realvalue = quad(lambda x: np.real(g(x,k,n)), a, b, epsabs = 1e-8)[0]
    imagvalue = quad(lambda x: np.imag(g(x,k,n)), a, b, epsabs = 1e-8)[0]
    value = realvalue + 1.j*imagvalue
    return [value]



def complex_integral(g, rect, k, integr = quad_integral, n=0):
    """Contour integral on a box of size L"""
    x0, y0, x1, y1 = rect

    def real_integrand(x, k, n):
        g1 = g(x + 1j*y0, k, n)
        g2 = g(x + 1j*y1, k, n)
        return  g1 - g2

    def imaginary_integrand(y, k, n):
        return g(x1 + 1j*y, k, n)- g(x0 + 1j*y, k, n)

    integral1 = integr(real_integrand, x0, x1, k, n )
    integral2 = integr(imaginary_integrand, y0, y1, k, n)

    # val = r1 - I2 + j ( I1 + R2) = 1 + j 2
    val = integral1[0] + 1j*integral2[0]

    return val




def re_center(rect, new_center):
    """
    return a rectangle of the same length as rect
    but centered around a new center
    """
    length = rect[2] - rect[0]
    height = rect[3] - rect[1]

    x0 = new_center[0] - length * 0.5
    x1 = x0 + length
    y0 = new_center[1] - height * 0.5
    y1 = y0 + height

    return (x0, y0, x1, y1)



def grow(rect, coef):
    """
    return a rectangle of similar center
    but length and heigth multiplied by coef
    """
    length = rect[2] - rect[0]
    height = rect[3] - rect[1]

    center = (rect[0] + 0.5 * length, rect[1] + 0.5 * height)
    new_length = coef * length
    new_height = coef * height
    x0 = center[0] - new_length * 0.5
    x1 = x0 + new_length
    y0 = center[1] - new_height * 0.5
    y1 = y0 + new_height

    return (x0, y0, x1, y1)



def dist_cpx(c1, c2):
    """
    returns the distance between two complex numbers
    """
    return np.abs(np.abs(c1) - np.abs(c2))





class SolverNode(Node):

    def __init__(self, parent, domain, k, function, max_depth,
                 deriv_function = None):
        self.rect = domain
        self.max_depth = max_depth
        self.function = function
        self.has_zero = False
        self.k = k
        self.parent = parent
        self.deriv_function = deriv_function
        self.error_domains = None
        self.errors = None
        self.precision = None
        super().__init__(self.parent, domain)


    def getinstance(self, rect):

        return SolverNode(self, rect,
                          self.k,
                          self.function,
                          self.max_depth,
                          deriv_function=self.deriv_function)


    def has_converged(self):
        return self.has_zero



    def integ(self, rect, n=0, integr_function = raw_integral):
        # TODO big shit happens if deriv_function is None

        if self.deriv_function is not None:

            def integrand_func(z,k,n=n):
                ret = -1j*0.5/np.pi*z**n*self.deriv_function(z, k)/self.function(z, k)
                return ret

            return complex_integral(integrand_func, rect, self.k, integr=integr_function, n=n)

        else:
            if integr_function == raw_integral:
                return complex_integral_num_raw(self.function, rect, self.k, n=n)

            return complex_integral_num(self.function, rect, self.k, integr=integr_function, n=n)




    def suspicious_zero(self):
        error =   self.precision[0]
        round_val = self.precision[1]
        val = self.precision[2]
        return (error > 0.01 and round_val <= 1) or ( val < -0.01)


    def no_zero(self):
        round_val = self.precision[1]
        return round_val < 1


    def found_zero(self):
        round_val = self.precision[1]
        error =   self.precision[0]
        return round_val == 1 and error < 0.01


    def growth(self,rect):
        """return a slighly bigger rectange"""
        x0, y0, x1, y1 = rect
        factor = 1e-4
        Ly, Lx = y1 - y0, x1 - x0

        return [ x0 - factor*Lx, y0 - factor*Ly, x1 + factor*Lx, y1 + factor*Ly ]



    def needs_subdivision(self, rect):

        subdivide = True
        dont_subdivide = False

        if self.depth == self.max_depth:
           # self.has_zero = True
            return dont_subdivide

        val = self.integ(rect, integr_function=raw_integral)
        if np.isnan(np.real(val)):
            return dont_subdivide

        round_val = int(np.round(np.real(val)))
        error = abs(round_val - val.real)

        self.precision = (error, round_val, val.real)

        if self.suspicious_zero():
            rect = self.growth(rect)
            self.rect = rect
            val = self.integ(rect, integr_function=quad_integral)
            round_val = int(np.round(np.real(val)))
            error = abs(round_val - val.real)
            self.precision = (error, round_val, val.real)


        if self.suspicious_zero():
            return subdivide

        if self.no_zero():
            self.has_zero = False
            return dont_subdivide

        if self.found_zero():
            self.has_zero = True
            return dont_subdivide

        else:
            return subdivide



class Solver(QuadTree):

    def __init__(self, function, domain, k, max_depth=10, deriv_function=None):


        self.max_depth = max_depth
        self.function = function
        self.deriv_function = deriv_function
        self.k = k

        self.domain = domain

        self.root = SolverNode(None, self.domain, self.k, self.function,
                               self.max_depth, deriv_function=self.deriv_function)



    def solve(self, methode = "integral"):

        super().__init__(self.root, self.max_depth)

        if methode == "integral":
            self.zeros = np.asarray([z.integ(z.rect, n=1, integr_function=quad_integral) for z in self.leaves])
            for i,leave in enumerate(self.leaves):
                leave.zero = self.zeros[i]

        elif methode == "cg":

            wrap = gs.wrapper(self.function, k=self.k)

            self.zeros = np.zeros(len(self.leaves), dtype="c8")

            for i, leave in enumerate(self.leaves):
                x0 = [(leave.rect[0] + leave.rect[2]) / 2, (leave.rect[1] + leave.rect[3]) / 2]
                bounds = [[leave.rect[0], leave.rect[2]], [leave.rect[1], leave.rect[3]]]

                v = wrap.solve(x0, bounds=bounds)[0]

                self.zeros[i] = v[0] + 1.j*v[1]
                leave.zero = self.zeros[i]




    def clean_double_roots(self):
        """remove the doubles roots"""
        tmp = self.zeros.copy()
        realtmp = np.real(tmp)
        imagtmp = np.imag(tmp)

        indexsort = np.argsort(realtmp)

        realtmp = realtmp[indexsort]
        imagtmp = imagtmp[indexsort]

        d = np.append(True, np.abs(np.diff(realtmp)))
        e = np.append(True, np.abs(np.diff(imagtmp)))
        TOL = 1e-5

        tmp = tmp[indexsort]
        condition = [(i > TOL) and (j > TOL) for i, j in zip(d, e)]
        tmp = tmp[condition]
        self.zeros = tmp

        self.leaves = [ self.leaves[i] for i in indexsort]
        new_leaves = []
        for i, cond in enumerate(condition):
            if cond:
                new_leaves.append(self.leaves[i])

        self.leaves = new_leaves



    def precision_global(self):
        precision = 0.
        max_error = 0.
        for node in self.allnodes:
            if node.precision[1] == 1:
                precision = node.precision[0]**2
                if np.abs(node.precision[0]) > max_error:
                    max_error = np.abs(node.precision[0])

        return np.sqrt(precision)/len(self.allnodes), max_error



    def precision(self):
        contour_diff_a = np.zeros_like(self.zeros, dtype=np.float64)
        distances = np.zeros(self.zeros.size)

        for i, leave in enumerate(self.leaves):
            rect = leave.rect
            zero = leave.zero

            new_rect = re_center(rect, (zero.real, zero.imag))
            new_rect = grow(new_rect, 0.02)

            nbr_after_r = np.real(leave.integ(new_rect, n=0, integr_function=quad_integral))
            nbr_after = int(np.round(nbr_after_r))

            new_zero = leave.integ(new_rect, n=1, integr_function=quad_integral)

            distances[i] = dist_cpx(zero, new_zero)
            contour_diff_a[i] = nbr_after_r

        res = np.sum(distances) / distances.size

        rounds = np.round(contour_diff_a)
        return res, rounds[rounds != 1]


def grid(w_domain, N):
    wr_min = w_domain[0]
    wi_min = w_domain[1]

    wr_max = w_domain[2]
    wi_max = w_domain[3]

    # let's resample this...
    dwr = (wr_max - wr_min) / (N - 1)
    nwr = N
    wr = np.arange(nwr) * dwr + wr_min

    dwi = (wi_max - wi_min) / (N - 1)
    nwi = N
    wi = np.arange(nwi) * dwi + wi_min

    # ... and build a multidimensional version of w
    wwr, wwi = np.meshgrid(wr, wi, indexing='ij')
    ww = wwr + 1j * wwi

    return ww





def plot_tree(tree, function, w_domain, k, N = 512, filename=None):
    """Plot the functions, the rectangles and the solutions"""

    ww = grid(w_domain, N)
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    # ax.imshow(np.abs(values), origin='lower',vmin=0, vmax=1)

    values = function(ww, k)
    im = ax.pcolormesh(ww.real, ww.imag, np.abs(values), vmin=0, vmax=1)  # , vmax=0.5)

    ax.set_title("depth = %d" % (tree.max_depth))
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")

    ax.set_xlim((w_domain[0], w_domain[2]))
    ax.set_ylim((w_domain[1], w_domain[3]))

    for z in tree.allnodes:
        r = z.rect
        # print("node ; ",z.has_zero)
        rect = Rectangle((r[0],r[1]), (r[2]-r[0]), (r[3]-r[1]), ec="k", fc="none", ls='--', lw=0.2)
        ax.add_patch(rect)

    for z in tree.leaves:
        # print(z.has_zero)
        if z.has_zero:
            r = z.rect
            rect = Rectangle((r[0], r[1]), (r[2] - r[0]), (r[3] - r[1]), ec="r", fc="none", ls='-', lw=1)
            ax.add_patch(rect)


    for node in tree.allnodes:
        if node.error_domains is not None:
            r =  node.error_domains
            rect = Rectangle((r[0], r[1]), (r[2] - r[0]), (r[3] - r[1]), ec="w", fc="none", ls='--', lw=1)
            ax.add_patch(rect)


    ax.scatter(np.real(tree.zeros), np.imag(tree.zeros), marker='+', c='k')

    plt.colorbar(im, ax=ax)

    if filename is not None:
        f.savefig(filename)
