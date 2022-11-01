from utils import check_num_alike
import numpy as np
import pandas as pd

from findiff import FinDiff
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line

METHOD_NAIVE, METHOD_NAIVECONSEC, METHOD_NUMDIFF = 0, 1, 2
METHOD_NCUBED, METHOD_NSQUREDLOGN, METHOD_HOUGHPOINTS, METHOD_HOUGHLINES, METHOD_PROBHOUGH = 0, 1, 2, 3, 4

def get_extrema(h, extmethod=METHOD_NUMDIFF, accuracy=1):
    #h must be single dimensional array-like object e.g. List, np.ndarray, pd.Series
    if type(h) is tuple and len(h) == 2 and (h[0] is None or check_num_alike(h[0])) and (h[1] is None or check_num_alike(h[1])) and (not h[0] is None or not h[1] is None):
        hmin, hmax = h[0], h[1]
        if not h[0] is None and not h[1] is None and len(hmin) != len(hmax): #not strict requirement, but contextually ideal
            raise ValueError('h does not have a equal length minima and maxima data')
    elif check_num_alike(h): hmin, hmax = None, None
    else: raise ValueError('h is not list, numpy ndarray or pandas Series of numeric values or a 2-tuple thereof')
    if extmethod == METHOD_NAIVE:
        #naive method
        def get_minmax(h):
            rollwin = pd.Series(h).rolling(window=3, min_periods=1, center=True)
            minFunc = lambda x: len(x) == 3 and x.iloc[0] > x.iloc[1] and x.iloc[2] > x.iloc[1]
            maxFunc = lambda x: len(x) == 3 and x.iloc[0] < x.iloc[1] and x.iloc[2] < x.iloc[1]
            numdiff_extrema = lambda func: np.flatnonzero(rollwin.aggregate(func)).tolist()
            return minFunc, maxFunc, numdiff_extrema            
    elif extmethod == METHOD_NAIVECONSEC:
        #naive method collapsing duplicate consecutive values
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
        dx = 1 #1 day interval
        d_dx = FinDiff(0, dx, 1, acc=accuracy) #acc=3 #for 5-point stencil, currenly uses +/-1 day only
        d2_dx2 = FinDiff(0, dx, 2, acc=accuracy) #acc=3 #for 5-point stencil, currenly uses +/-1 day only
        def get_minmax(h):
            clarr = np.asarray(h, dtype=np.float64)
            mom, momacc = d_dx(clarr), d2_dx2(clarr)
            #print(mom[-10:], momacc[-10:])
            #numerical derivative will yield prominent extrema points only
            def numdiff_extrema(func):
                return [x for x in range(len(mom))
                        if func(x) and
                            (mom[x] == 0 or #either slope is 0, or it crosses from positive to negative with the closer to 0 of the two chosen or prior if a tie
                             (x != len(mom) - 1 and (mom[x] > 0 and mom[x+1] < 0 and h[x] >= h[x+1] or #mom[x] >= -mom[x+1]
                                                     mom[x] < 0 and mom[x+1] > 0 and h[x] <= h[x+1]) or #-mom[x] >= mom[x+1]) or
                              x != 0 and (mom[x-1] > 0 and mom[x] < 0 and h[x-1] < h[x] or #mom[x-1] < -mom[x] or
                                          mom[x-1] < 0 and mom[x] > 0 and h[x-1] > h[x])))] #-mom[x-1] < mom[x])))]
            return lambda x: momacc[x] > 0, lambda x: momacc[x] < 0, numdiff_extrema
    else: raise ValueError('extmethod must be METHOD_NAIVE, METHOD_NAIVECONSEC, METHOD_NUMDIFF')
    if hmin is None and hmax is None:
        minFunc, maxFunc, numdiff_extrema = get_minmax(h)
        return numdiff_extrema(minFunc), numdiff_extrema(maxFunc)
    if not hmin is None:
        minf = get_minmax(hmin)
        if hmax is None: return minf[2](minf[0])
    if not hmax is None:
        maxf = get_minmax(hmax)
        if hmin is None: return maxf[2](maxf[1])
    return minf[2](minf[0]), maxf[2](maxf[1])





def calc_support_resistance(h, extmethod = METHOD_NUMDIFF, method=METHOD_NSQUREDLOGN,
                            window=125, errpct=0.005, hough_scale=0.01, hough_prob_iter=10,
                            sortError=False, accuracy=1):
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
    #h = hist.Close.tolist()
    if type(h) is tuple and len(h) == 2 and (h[0] is None or check_num_alike(h[0])) and (h[1] is None or check_num_alike(h[1])) and (not h[0] is None or not h[1] is None):
        if not h[0] is None and not h[1] is None and len(h[0]) != len(h[1]): #not strict requirement, but contextually ideal
            raise ValueError('h does not have a equal length minima and maxima data')
        hmin, hmax, len_h = h[0], h[1], len(h[1 if h[0] is None else 0])
    elif check_num_alike(h): hmin, hmax, len_h = None, None, len(h)
    else: 
        raise ValueError('h is not list, numpy ndarray or pandas Series of numeric values or a 2-tuple thereof')
    def get_bestfit3(x0, y0, x1, y1, x2, y2):
        xbar, ybar = (x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3
        xb0, yb0, xb1, yb1, xb2, yb2 = x0-xbar, y0-ybar, x1-xbar, y1-ybar, x2-xbar, y2-ybar
        xs = xb0*xb0+xb1*xb1+xb2*xb2
        m = (xb0*yb0+xb1*yb1+xb2*yb2) / xs
        b = ybar - m * xbar
        ys0, ys1, ys2 = (y0 - (m * x0 + b)),(y1 - (m * x1 + b)),(y2 - (m * x2 + b))
        ys = ys0*ys0+ys1*ys1+ys2*ys2
        ser = np.sqrt(ys / xs)
        return m, b, ys, ser, ser * np.sqrt((x0*x0+x1*x1+x2*x2)/3)
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
    def get_trend(Idxs, h, fltpct, min_h, max_h):
        trend = []
        for x in range(len(Idxs)): #unfortunately an O(n(n-1)(n-2))=O((n^2-n)(n-2))=O(n^3-3n^2-2n)~=O(n^3) algorithm but meets the strict definition of a trendline
            for y in range(x+1, len(Idxs)):
                #slope = (h[Idxs[x]] - h[Idxs[y]]) / (Idxs[x] - Idxs[y]) #m=dy/dx #if slope 0 then intercept does not exist constant y where y=b
                #intercept = h[Idxs[x]] - slope * Idxs[x] #y=mx+b, b=y-mx
                for z in range(y+1, len(Idxs)):
                    #distance = abs(slope * Idxs[z] + intercept - h[Idxs[z]]) #distance to y value based on x with slope-intercept
                    trend.append(([Idxs[x], Idxs[y], Idxs[z]], get_bestfit3(Idxs[x], h[Idxs[x]], Idxs[y], h[Idxs[y]], Idxs[z], h[Idxs[z]])))
        return list(filter(lambda val: val[1][3] <= fltpct, trend))
    def get_trend_opt(Idxs, h, fltpct, min_h, max_h):
        slopes, trend = [], []
        for x in range(len(Idxs)): #O(n^2*log n) algorithm
            slopes.append([])
            for y in range(x+1, len(Idxs)):
                slope = (h[Idxs[x]] - h[Idxs[y]]) / (Idxs[x] - Idxs[y]) #m=dy/dx #if slope 0 then intercept does not exist constant y where y=b
                #intercept = h[Idxs[x]] - slope * Idxs[x] #y=mx+b, b=y-mx
                slopes[x].append((slope, y))
        for x in range(len(Idxs)):
            slopes[x].sort() #key=lambda val: val[0])
            CurIdxs = [Idxs[x]]
            for y in range(0, len(slopes[x])):
                #distance = abs(slopes[x][y][2] * slopes[x][y+1][1] + slopes[x][y][3] - h[slopes[x][y+1][1]])
                CurIdxs.append(Idxs[slopes[x][y][1]])
                if len(CurIdxs) < 3: continue
                res = get_bestfit([(p, h[p]) for p in CurIdxs])
                if res[3] <= fltpct:
                    CurIdxs.sort()
                    if len(CurIdxs) == 3:
                        trend.append((CurIdxs, res))
                        CurIdxs = list(CurIdxs)
                    else: CurIdxs, trend[-1] = list(CurIdxs), (CurIdxs, res)
                    #if len(CurIdxs) >= MaxPts: CurIdxs = [CurIdxs[0], CurIdxs[-1]]
                else: CurIdxs = [CurIdxs[0], CurIdxs[-1]] #restart search from this point
        return trend
    def make_image(Idxs, h, min_h, max_h):
        #np.arctan(2/len_h), np.arctan(2/int((hist.Close.max() - m + 1) * (1/hough_scale))) #minimal angles to find all points
        max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
        m, tested_angles = min_h, np.linspace(-np.pi / 2, np.pi / 2, 360*5) #degree of precision from 90 to 270 degrees with 360*5 increments
        height = int((max_h - m + 0.01) * (1/hough_scale))
        mx = min(max_size, height)
        scl = (1/hough_scale) * mx / height
        image = np.zeros((mx, len_h)) #in rows, columns or y, x image format
        for x in Idxs:
            image[int((h[x] - m) * scl), x] = 255
        return image, tested_angles, scl, m
    def find_line_pts(Idxs, x0, y0, x1, y1, h, fltpct):
        s = (y0 - y1) / (x0 - x1)
        i, dnm = y0 - s * x0, np.sqrt(1 + s*s)
        dist = [(np.abs(i+s*x-h[x])/dnm, x) for x in Idxs]
        dist.sort() #(key=lambda val: val[0])
        pts, res = [], None
        for x in range(len(dist)):
            pts.append((dist[x][1], h[dist[x][1]]))
            if len(pts) < 3: continue
            r = get_bestfit(pts)
            if r[3] > fltpct:
                pts = pts[:-1]
                break
            res = r
        pts = [x for x, _ in pts]
        pts.sort()
        return pts, res
    def hough_points(pts, width, height, thetas):
        diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
        rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
        # Cache some resuable values
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)
        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
        # Vote in the hough accumulator
        for i in range(len(pts)):
            x, y = pts[i]
            for t_idx in range(num_thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_len
                accumulator[rho, t_idx] += 1
        return accumulator, thetas, rhos
    def houghpt(Idxs, h, fltpct, min_h, max_h):
        max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
        m, tested_angles = min_h, np.linspace(-np.pi / 2, np.pi / 2, 360*5) #degree of precision from 90 to 270 degrees with 360*5 increments
        height = int((max_h - m + 1) * (1/hough_scale))
        mx = min(max_size, height)
        scl = (1/hough_scale) * mx / height
        acc, theta, d = hough_points([(x, int((h[x] - m) * scl)) for x in Idxs], mx, len_h, np.linspace(-np.pi / 2, np.pi / 2, 360*5))
        origin, lines = np.array((0, len_h)), []
        for x, y in np.argwhere(acc >= 3):
            dist, angle = d[x], theta[y]
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            y0, y1 = y0 / scl + m, y1 / scl + m
            pts, res = find_line_pts(Idxs, 0, y0, len_h, y1, h, fltpct)
            if len(pts) >= 3: lines.append((pts, res))
        return lines
    def hough(Idxs, h, fltpct, min_h, max_h):
        image, tested_angles, scl, m = make_image(Idxs, h, min_h, max_h)
        hl, theta, d = hough_line(image, theta=tested_angles)
        origin, lines = np.array((0, image.shape[1])), []
        for pts, angle, dist in zip(*hough_line_peaks(hl, theta, d, threshold=2)): #> threshold
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            y0, y1 = y0 / scl + m, y1 / scl + m
            pts, res = find_line_pts(Idxs, 0, y0, image.shape[1], y1, h, fltpct)
            if len(pts) >= 3: lines.append((pts, res))
        return lines
    def prob_hough(Idxs, h, fltpct, min_h, max_h):
        image, tested_angles, scl, m = make_image(Idxs, h, min_h, max_h)
        lines = []
        for x in range(hough_prob_iter):
            lines.extend(probabilistic_hough_line(image, threshold=2, theta=tested_angles, line_length=0,
                                            line_gap=int(np.ceil(np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1]))))))
        l = []
        for (x0, y0), (x1, y1) in lines:
            if x0 == x1: continue
            if x1 < x0: (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
            y0, y1 = y0 / scl + m, y1 / scl + m
            pts, res = find_line_pts(Idxs, x0, y0, x1, y1, h, fltpct)
            if len(pts) >= 3: l.append((pts, res))
        return l
    def merge_lines(Idxs, trend, h, fltpct):
        for x in Idxs:
            l = []
            for i, (p, r) in enumerate(trend):
                if x in p: l.append((r[0], i))
            l.sort() #key=lambda val: val[0])
            if len(l) > 1: CurIdxs = list(trend[l[0][1]][0])
            for (s, i) in l[1:]:
                CurIdxs += trend[i][0]
                CurIdxs = list(dict.fromkeys(CurIdxs))
                CurIdxs.sort()
                res = get_bestfit([(p, h[p]) for p in CurIdxs])
                if res[3] <= fltpct: trend[i-1], trend[i], CurIdxs = ([], None), (CurIdxs, res), list(CurIdxs)
                else: CurIdxs = list(trend[i][0]) #restart search from here
        return list(filter(lambda val: val[0] != [], trend))
    def measure_area(trendline, isMin, h): # Reimann sum of line to discrete time series data
        #first determine the time range, and subtract the line values to obtain a single function
        #support subtracts the line minus the series and eliminates the negative values
        #resistances subtracts the series minus the line and eliminate the negatives
        base = trendline[0][0]
        m, b, ser = trendline[1][0], trendline[1][1], h[base:trendline[0][-1]+1]
        return sum([max(0, (m * (x+base) + b) - y if isMin else y - (m * (x+base) + b)) for x, y in enumerate(ser)]) / len(ser)
    def window_results(trends, isMin, h):
        windows = [[] for x in range(len(divide)-1)]
        for x in trends:
            fstwin, lastwin = int(x[0][0] / window), int(x[0][-1] / window)
            wins = [[] for _ in range(fstwin, lastwin+1)]
            for y in x[0]: wins[int(y / window) - fstwin].append(y)
            for y in range(0, lastwin-fstwin):
                if len(wins[y+1]) == 0 and len(wins[y]) >= 3: windows[fstwin+y].append(wins[y])
                if len(wins[y]) + len(wins[y + 1]) >= 3:
                    windows[fstwin+y+1].append(wins[y] + wins[y+1])
            if lastwin-fstwin==0 and len(wins[0]) >= 3: windows[fstwin].append(wins[0])
        def fitarea(x):
            fit = get_bestfit([(y, h[y]) for y in x])
            return (x, fit + (measure_area((x, fit), isMin, h),))
        def dosort(x):
            x.sort(key = lambda val: val[1][skey])
            return x
        return [dosort(list(fitarea(pts) for pts in x)) for x in windows]
    #print((mintrend[:5], maxtrend[:5]))
    
    #find all places where derivative is 0 - in finite case when it crosses positive to negative and choose the closer to 0 value
    #second derivative being positive or negative decides if they are minima or maxima
    #now for all pairs of 3 points construct the average line, rate it based on # of additional points, # of points on the wrong side of the line, and the margin of error for the line passing through all of them
    #finally select the best based on this rating

    #first find the peaks and troughs
    
    def overall_line(idxs, vals):
        if len(idxs) <= 1: pm, zme = [np.nan, np.nan], [np.nan]
        else:
            p, r = np.polynomial.polynomial.Polynomial.fit(idxs, vals, 1, full=True) #more numerically stable
            pm, zme = list(reversed(p.convert().coef)), r[0]
            if len(pm) == 1: pm.insert(0, 0.0)
        return pm  
    def calc_all(idxs, h, isMin):
        min_h, max_h = min(h), max(h)
        scale = (max_h - min_h) / len_h
        fltpct = scale * errpct
        midxs = [[] for _ in range(len(divide)-1)]
        for x in idxs: midxs[int((x + rem) / window)].append(x)
        mtrend = []
        for x in range(len(divide)-1-1):
            m = midxs[x] + midxs[x+1]
            mtrend.extend(trendmethod(m, h, fltpct, min_h, max_h))
        if len(divide) == 2:
            mtrend.extend(trendmethod(midxs[0], h, fltpct, min_h, max_h))
        mtrend = merge_lines(idxs, mtrend, h, fltpct)
        mtrend = [(pts, (res[0], res[1], res[2], res[3], res[4], measure_area((pts, res), isMin, h))) for pts, res in mtrend]
        mtrend.sort(key=lambda val: val[1][skey])
        mwindows = window_results(mtrend, isMin, h)
        pm = overall_line(idxs, [h[x] for x in idxs])
        #print((pmin, pmax, zmne, zmxe))
        return pm, mtrend, mwindows
    if method == METHOD_NCUBED:
        trendmethod = get_trend
    elif method == METHOD_NSQUREDLOGN:
        trendmethod = get_trend_opt
    elif method == METHOD_HOUGHPOINTS:
        trendmethod = houghpt
    #pip install scikit-image
    elif method == METHOD_HOUGHLINES:
        trendmethod = hough
    elif method == METHOD_PROBHOUGH:
        trendmethod = prob_hough
    else: raise ValueError('method must be one of METHOD_NCUBED, METHOD_NSQUREDLOGN, METHOD_HOUGHPOINTS, METHOD_HOUGHLINES, METHOD_PROBHOUGH')
    extremaIdxs = get_extrema(h, extmethod, accuracy)
    divide = list(reversed(range(len_h, -window, -window)))
    rem, divide[0] = window - len_h % window, 0
    if rem == window: rem = 0
    skey = 3 if sortError else 5
    if hmin is None and hmax is None:
        pmin, mintrend, minwindows = calc_all(extremaIdxs[0], h, True)
        pmax, maxtrend, maxwindows = calc_all(extremaIdxs[1], h, False)
    else:
        if not hmin is None:
            pmin, mintrend, minwindows = calc_all(extremaIdxs if hmax is None else extremaIdxs[0], hmin, True)
            if hmax is None: return (extremaIdxs, pmin, mintrend, minwindows)
        if not hmax is None:            
            pmax, maxtrend, maxwindows = calc_all(extremaIdxs if hmin is None else extremaIdxs[1], hmax, False)
            if hmin is None: return (extremaIdxs, pmax, maxtrend, maxwindows)
    return (extremaIdxs[0], pmin, mintrend, minwindows), (extremaIdxs[1], pmax, maxtrend, maxwindows)