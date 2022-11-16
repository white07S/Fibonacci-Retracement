from src.codeReview.utils import check_num_alike

from src.codeReview.sup_res.methods.methods import *
from src.codeReview.sup_res.extrema.extrema import *
from src.codeReview.sup_res.calculation.calculation import calc_all


class SupResFinder:
    def __init__(self):
        pass

    def validate_input(self, extmethod, method,
        window, errpct, hough_scale, hough_prob_iter,
        sortError, accuracy):
        """Raise an error if some argument is of an incorrect type"""
        # TODO - type hints in the respective functions
        if not type(window) is int:
            raise ValueError('window must be of type int')
        if not type(errpct) is float:
            raise ValueError('errpct must be of type float')
        if not type(hough_scale) is float:
            raise ValueError('house_scale must be of type float')
        if not type(hough_prob_iter) is int:
            raise ValueError('house_prob_iter must be of type int')
        if not type(sortError) is bool:
            raise ValueError('sortError must be True of False')
            # h = hist.Close.tolist()

    def read_h(self, h):
        """
        Check whether `h` is valid data, if it is, then return tuple containing
        maximal records, minimal records, and length
        """
        # FIXME - duplicates should be separate functions
        if type(h) is tuple and len(h) == 2 and (h[0] is None or check_num_alike(h[0])) and (
                h[1] is None or check_num_alike(h[1])) and (not h[0] is None or not h[1] is None):
            if not h[0] is None and not h[1] is None and len(h[0]) != len(
                    h[1]):  # not strict requirement, but contextually ideal
                raise ValueError('h does not have a equal length minima and maxima data')
            hmin, hmax, len_h = h[0], h[1], len(h[1 if h[0] is None else 0])
        elif check_num_alike(h):
            hmin, hmax, len_h = None, None, len(h)
        else:
            raise ValueError('h is not list, numpy ndarray or pandas Series of numeric values or a 2-tuple thereof')
        
        return hmin, hmax, len_h

    def calc_support_resistance(self,
        h, extmethod=METHOD_NUMDIFF, method=METHOD_NSQUREDLOGN,
        window=125, errpct=0.005, hough_scale=0.01, hough_prob_iter=10,
        sortError=False, accuracy=1):

        self.validate_input(extmethod, method, window, errpct, hough_scale, hough_prob_iter,sortError, accuracy)

        hmin, hmax, len_h = self.read_h(h)
        trendmethod = get_method(method)
        extremaIdxs = get_extrema(h, extmethod, accuracy)

        if hmin is None and hmax is None:
            pmin, mintrend, minwindows = calc_all(extremaIdxs[0], h, True, len_h, errpct, window, trendmethod, sortError)
            pmax, maxtrend, maxwindows = calc_all(extremaIdxs[1], h, False, len_h, errpct, window, trendmethod, sortError)
        else:
            if not hmin is None:
                pmin, mintrend, minwindows = calc_all(extremaIdxs if hmax is None else extremaIdxs[0], hmin, True, len_h, errpct, window, trendmethod, sortError)
                if hmax is None:
                    return (extremaIdxs, pmin, mintrend, minwindows)
            if not hmax is None:
                pmax, maxtrend, maxwindows = calc_all(extremaIdxs if hmin is None else extremaIdxs[1], hmax, False, len_h, errpct, window, trendmethod, sortError)
                if hmin is None:
                    return (extremaIdxs, pmax, maxtrend, maxwindows)

        return (extremaIdxs[0], pmin, mintrend, minwindows), (extremaIdxs[1], pmax, maxtrend, maxwindows)
