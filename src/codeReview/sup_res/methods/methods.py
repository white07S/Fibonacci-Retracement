from src.codeReview.sup_res.methods.hough_methods import *
from src.codeReview.sup_res.methods.trend_methods import *


METHOD_NCUBED, METHOD_NSQUREDLOGN, METHOD_HOUGHPOINTS, METHOD_HOUGHLINES, METHOD_PROBHOUGH = 0, 1, 2, 3, 4


def get_method(method):
    if method == METHOD_NCUBED:
        trendmethod = get_trend
    elif method == METHOD_NSQUREDLOGN:
        trendmethod = get_trend_opt
    elif method == METHOD_HOUGHPOINTS:
        trendmethod = houghpt
    # pip install scikit-image
    elif method == METHOD_HOUGHLINES:
        trendmethod = hough
    elif method == METHOD_PROBHOUGH:
        trendmethod = prob_hough
    else:
        raise ValueError('method must be one of METHOD_NCUBED, METHOD_NSQUREDLOGN, METHOD_HOUGHPOINTS, METHOD_HOUGHLINES, METHOD_PROBHOUGH')
    return trendmethod