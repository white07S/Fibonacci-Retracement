import pandas as pd
import numpy as np


def check_num_alike(h):
    """Return `True` if `h` consists of numbers only"""
    if type(h) is list and all([isinstance(x, (bool, int, float)) for x in h]):
        return True
    elif type(h) is np.ndarray and h.ndim == 1 and h.dtype.kind in 'biuf':
        return True
    else:
        if type(h) is pd.Series and h.dtype.kind in 'biuf':
            return True
        else:
            return False


# FIXME - handle get_bestfit within one function?
def get_bestfit3(x0, y0, x1, y1, x2, y2):
    xbar, ybar = (x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3
    xb0, yb0, xb1, yb1, xb2, yb2 = x0 - xbar, y0 - ybar, x1 - xbar, y1 - ybar, x2 - xbar, y2 - ybar
    xs = xb0 * xb0 + xb1 * xb1 + xb2 * xb2
    m = (xb0 * yb0 + xb1 * yb1 + xb2 * yb2) / xs
    b = ybar - m * xbar
    ys0, ys1, ys2 = (y0 - (m * x0 + b)), (y1 - (m * x1 + b)), (y2 - (m * x2 + b))
    ys = ys0 * ys0 + ys1 * ys1 + ys2 * ys2
    ser = np.sqrt(ys / xs)

    return m, b, ys, ser, ser * np.sqrt((x0 * x0 + x1 * x1 + x2 * x2) / 3)

def get_bestfit(pts):
    xbar, ybar = [sum(x) / len(x) for x in zip(*pts)]

    def subcalc(x, y):
        tx, ty = x - xbar, y - ybar
        return tx * ty, tx * tx, x * x

    (xy, xs, xx) = [sum(q) for q in zip(*[subcalc(x, y) for x, y in pts])]
    m = xy / xs
    b = ybar - m * xbar
    ys = sum([np.square(y - (m * x + b)) for x, y in pts])
    ser = np.sqrt(ys / ((len(pts) - 2) * xs))
    return m, b, ys, ser, ser * np.sqrt(xx / len(pts))
