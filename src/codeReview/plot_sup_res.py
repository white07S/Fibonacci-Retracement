from sup_res_preliminary import calc_support_resistance, METHOD_NUMDIFF, METHOD_NSQUREDLOGN
from utils import datefmt, check_num_alike

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_support_resistance(hist, xformatter = None, numbest = 2, fromwindows = True,
                            pctbound=0.1, extmethod = METHOD_NUMDIFF, method=METHOD_NSQUREDLOGN,
                            window=125, errpct = 0.005, hough_scale=0.01, hough_prob_iter=10, sortError=False, accuracy=1):
    
    ret = calc_support_resistance(hist, extmethod, method, window, errpct, hough_scale, hough_prob_iter, sortError, accuracy)
    plt.clf()
    plt.subplot(111)
    if len(ret) == 2:
        minimaIdxs, pmin, mintrend, minwindows = ret[0]
        maximaIdxs, pmax, maxtrend, maxwindows = ret[1]
        if type(hist) is tuple and len(hist) == 2 and check_num_alike(hist[0]) and check_num_alike(hist[1]):
            len_h = len(hist[0])
            min_h, max_h = min(min(hist[0]), min(hist[1])), max(max(hist[0]), max(hist[1]))
            disp = [(hist[0], minimaIdxs, pmin, 'yo', 'Avg. Support', 'y--'), (hist[1], maximaIdxs, pmax, 'bo', 'Avg. Resistance', 'b--')]
            dispwin = [(hist[0], minwindows, 'Support', 'g--'), (hist[1], maxwindows, 'Resistance', 'r--')]
            disptrend = [(hist[0], mintrend, 'Support', 'g--'), (hist[1], maxtrend, 'Resistance', 'r--')]
            plt.plot(range(len_h), hist[0], 'k--', label='Low Price')
            plt.plot(range(len_h), hist[1], 'm--', label='High Price')
        else:
            len_h = len(hist)
            min_h, max_h = min(hist), max(hist)
            disp = [(hist, minimaIdxs, pmin, 'yo', 'Avg. Support', 'y--'), (hist, maximaIdxs, pmax, 'bo', 'Avg. Resistance', 'b--')]
            dispwin = [(hist, minwindows, 'Support', 'g--'), (hist, maxwindows, 'Resistance', 'r--')]
            disptrend = [(hist, mintrend, 'Support', 'g--'), (hist, maxtrend, 'Resistance', 'r--')]
            plt.plot(range(len_h), hist, 'k--', label='Close Price')
    else:
        minimaIdxs, pmin, mintrend, minwindows = ([], [], [], []) if hist[0] is None else ret
        maximaIdxs, pmax, maxtrend, maxwindows = ([], [], [], []) if hist[1] is None else ret
        len_h = len(hist[1 if hist[0] is None else 0])
        min_h, max_h = min(hist[1 if hist[0] is None else 0]), max(hist[1 if hist[0] is None else 0])
        disp = [(hist[1], maximaIdxs, pmax, 'bo', 'Avg. Resistance', 'b--') if hist[0] is None else (hist[0], minimaIdxs, pmin, 'yo', 'Avg. Support', 'y--')]
        dispwin = [(hist[1], maxwindows, 'Resistance', 'r--') if hist[0] is None else (hist[0], minwindows, 'Support', 'g--')]
        disptrend = [(hist[1], maxtrend, 'Resistance', 'r--') if hist[0] is None else (hist[0], mintrend, 'Support', 'g--')]
        plt.plot(range(len_h), hist[1 if hist[0] is None else 0], 'k--', label= ('High' if hist[0] is None else 'Low') + ' Price')
    for h, idxs, pm, clrp, lbl, clrl in disp:
        plt.plot(idxs, [h[x] for x in idxs], clrp)
        plt.plot([0, len_h-1],[pm[1],pm[0] * (len_h-1) + pm[1]],clrl, label=lbl)
    def add_trend(h, trend, lbl, clr, bFirst):
        for ln in trend[:numbest]:
            maxx = ln[0][-1]+1
            while maxx < len_h:
                ypred = ln[1][0] * maxx + ln[1][1]
                if (h[maxx] > ypred and h[maxx-1] < ypred or h[maxx] < ypred and h[maxx-1] > ypred or
                    ypred > max_h + (max_h-min_h)*pctbound or ypred < min_h - (max_h-min_h)*pctbound): break
                maxx += 1
            x_vals = np.array((ln[0][0], maxx)) # plt.gca().get_xlim())
            y_vals = ln[1][0] * x_vals + ln[1][1]
            if bFirst:
                plt.plot([ln[0][0], maxx], y_vals, clr, label=lbl)
                bFirst = False
            else: plt.plot([ln[0][0], maxx], y_vals, clr)
        return bFirst
    if fromwindows:
        for h, windows, lbl, clr in dispwin:
            bFirst = True
            for trend in windows:
                bFirst = add_trend(h, trend, lbl, clr, bFirst)
    else:
        for h, trend, lbl, clr in disptrend:
            add_trend(h, trend, lbl, clr, True)
    plt.title('Prices with Support/Resistance Trend Lines')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(6))
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if not xformatter is None: 
        plt.gca().xaxis.set_major_formatter(xformatter)
    plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right')
    plt.savefig('sup_res.png')
    plt.show()
    return plt.gcf()


def plot_sup_res_date(hist, idx, numbest = 2, fromwindows = True, pctbound=0.1,
                      extmethod = METHOD_NUMDIFF, method=METHOD_NSQUREDLOGN, window=125,
                      errpct = 0.005, hough_scale=0.01, hough_prob_iter=10, sortError=False, accuracy=1):
    return plot_support_resistance(hist, ticker.FuncFormatter(datefmt(idx)), numbest, fromwindows,
                                   pctbound, extmethod, method, window, errpct, hough_scale, hough_prob_iter, sortError, accuracy)

import pandas as pd
import yfinance as yf

df = yf.download('amd',period="max",rounding=True)
# plot_support_resistance(df[-250:].Close, df[-250:].index,accuracy=8) RUN method
# this is why we needed datefmt function, you can run (RUN method to see actual error)
# TypeError: 'formatter' must be an instance of matplotlib.ticker.Formatter, not a pandas.core.indexes.datetimes.DatetimeIndex

plot_sup_res_date(df[-250:].Close, df[-250:].index,accuracy=8)