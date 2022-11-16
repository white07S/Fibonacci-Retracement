import numpy as np
import pandas as pd

from findiff import FinDiff

from src.codeReview.utils import check_num_alike

METHOD_NAIVE, METHOD_NAIVECONSEC, METHOD_NUMDIFF = 0, 1, 2


def get_extrema(h, extmethod=METHOD_NUMDIFF, accuracy=1):
    """
    TODO
    @param h: 1D array-like object, e.g. List, np.ndarray, pd.Series
    @param extmethod: TODO
    @param accuracy: TODO
    @return:
    """
    # FIXME - refactor validation
    if type(h) is tuple and len(h) == 2 and (h[0] is None or check_num_alike(h[0])) and (
            h[1] is None or check_num_alike(h[1])) and (not h[0] is None or not h[1] is None):
        hmin, hmax = h[0], h[1]
        if not h[0] is None and not h[1] is None and len(hmin) != len(
                hmax):  # not strict requirement, but contextually ideal
            raise ValueError('h does not have a equal length minima and maxima data')
    elif check_num_alike(h):
        hmin, hmax = None, None
    else:
        raise ValueError('h is not list, numpy ndarray or pandas Series of numeric values or a 2-tuple thereof')

    # FIXME - the below need to be refactored for sure... (Szymon)
    if extmethod == METHOD_NAIVE:
        # naive method
        def get_minmax(h):
            rollwin = pd.Series(h).rolling(window=3, min_periods=1, center=True)
            minFunc = lambda x: len(x) == 3 and x.iloc[0] > x.iloc[1] and x.iloc[2] > x.iloc[1]
            maxFunc = lambda x: len(x) == 3 and x.iloc[0] < x.iloc[1] and x.iloc[2] < x.iloc[1]
            numdiff_extrema = lambda func: np.flatnonzero(rollwin.aggregate(func)).tolist()

            return minFunc, maxFunc, numdiff_extrema
    elif extmethod == METHOD_NAIVECONSEC:
        # naive method collapsing duplicate consecutive values
        def get_minmax(h):
            hist = pd.Series(h)
            rollwin = hist.loc[hist.shift(-1) != hist].rolling(window=3, center=True)
            minFunc = lambda x: x.iloc[0] > x.iloc[1] and x.iloc[2] > x.iloc[1]
            maxFunc = lambda x: x.iloc[0] < x.iloc[1] and x.iloc[2] < x.iloc[1]

            def numdiff_extrema(func):
                x = rollwin.aggregate(func)
                return x[x == 1].index.tolist()

            return minFunc, maxFunc, numdiff_extrema
    elif extmethod == METHOD_NUMDIFF:
        # FIXME - odd accuracy
        dx = 1  # 1 day interval
        d_dx = FinDiff(0, dx, 1, acc=accuracy)  # acc=3 #for 5-point stencil, currenly uses +/-1 day only
        d2_dx2 = FinDiff(0, dx, 2, acc=accuracy)  # acc=3 #for 5-point stencil, currenly uses +/-1 day only

        def get_minmax(h):
            clarr = np.asarray(h, dtype=np.float64)
            mom, momacc = d_dx(clarr), d2_dx2(clarr)

            # print(mom[-10:], momacc[-10:])
            # numerical derivative will yield prominent extrema points only

            def numdiff_extrema(func):
                # FIXME >:(
                return [x for x in range(len(mom))
                        if func(x) and
                        (mom[
                             x] == 0 or  # either slope is 0, or it crosses from positive to negative with the closer to 0 of the two chosen or prior if a tie
                         (x != len(mom) - 1 and (
                                 mom[x] > 0 and mom[x + 1] < 0 and h[x] >= h[x + 1] or  # mom[x] >= -mom[x+1]
                                 mom[x] < 0 and mom[x + 1] > 0 and h[x] <= h[x + 1]
                         ) or  # -mom[x] >= mom[x+1]) or
                          x != 0 and (
                                  mom[x - 1] > 0 and mom[x] < 0 and h[x - 1] < h[x] or  # mom[x-1] < -mom[x] or
                                  mom[x - 1] < 0 and mom[x] > 0 and h[x - 1] > h[x])))]  # -mom[x-1] < mom[x])))]

            return lambda x: momacc[x] > 0, lambda x: momacc[x] < 0, numdiff_extrema
    else:
        raise ValueError('extmethod must be METHOD_NAIVE, METHOD_NAIVECONSEC, METHOD_NUMDIFF')

    if hmin is None and hmax is None:
        minFunc, maxFunc, numdiff_extrema = get_minmax(h)
        return numdiff_extrema(minFunc), numdiff_extrema(maxFunc)
    if hmin is not None:
        minf = get_minmax(hmin)
        if hmax is None: return minf[2](minf[0])
    if hmax is not None:
        maxf = get_minmax(hmax)
        if hmin is None: return maxf[2](maxf[1])

    return minf[2](minf[0]), maxf[2](maxf[1])
