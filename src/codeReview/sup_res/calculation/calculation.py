import numpy as np
from src.codeReview.utils import get_bestfit



def merge_lines(Idxs, trend, h, fltpct):
    for x in Idxs:
        l = []
        for i, (p, r) in enumerate(trend):
            if x in p:
                l.append((r[0], i))
        l.sort()  # key=lambda val: val[0])
        if len(l) > 1:
            CurIdxs = list(trend[l[0][1]][0])
        for (s, i) in l[1:]:
            CurIdxs += trend[i][0]
            CurIdxs = list(dict.fromkeys(CurIdxs))
            CurIdxs.sort()
            res = get_bestfit([(p, h[p]) for p in CurIdxs])
            if res[3] <= fltpct:
                trend[i - 1], trend[i], CurIdxs = ([], None), (CurIdxs, res), list(CurIdxs)
            else:
                CurIdxs = list(trend[i][0])  # restart search from here

    return list(filter(lambda val: val[0] != [], trend))

def measure_area(trendline, isMin, h):
    """
    Reimann sum of line to discrete time series data
    - first determine the time range, and subtract the line values to obtain a single function
    - support subtracts the line minus the series and eliminates the negative values
    - resistances subtracts the series minus the line and eliminate the negatives
    @param trendline:
    @param isMin:
    @param h:
    @return:
    """
    base = trendline[0][0]
    m, b, ser = trendline[1][0], trendline[1][1], h[base:trendline[0][-1] + 1]

    return sum(
        [max(0, (m * (x + base) + b) - y if isMin else y - (m * (x + base) + b)) for x, y in enumerate(ser)]) / len(
        ser)

def window_results(trends, isMin, h, divide, window, skey):
    windows = [[] for x in range(len(divide) - 1)]
    for x in trends:
        fstwin, lastwin = int(x[0][0] / window), int(x[0][-1] / window)
        wins = [[] for _ in range(fstwin, lastwin + 1)]
        for y in x[0]: wins[int(y / window) - fstwin].append(y)
        for y in range(0, lastwin - fstwin):
            if len(wins[y + 1]) == 0 and len(wins[y]) >= 3:
                windows[fstwin + y].append(wins[y])
            if len(wins[y]) + len(wins[y + 1]) >= 3:
                windows[fstwin + y + 1].append(wins[y] + wins[y + 1])
        if lastwin - fstwin == 0 and len(wins[0]) >= 3:
            windows[fstwin].append(wins[0])

    def fitarea(x):
        fit = get_bestfit([(y, h[y]) for y in x])
        return x, fit + (measure_area((x, fit), isMin, h),)

    def dosort(x, skey):
        x.sort(key=lambda val: val[1][skey])
        return x

    return [dosort(list(fitarea(pts) for pts in x),skey) for x in windows]

# print((mintrend[:5], maxtrend[:5]))

# find all places where derivative is 0 - in finite case when it crosses positive to negative and choose the closer to 0 value
# second derivative being positive or negative decides if they are minima or maxima
# now for all pairs of 3 points construct the average line, rate it based on # of additional points, # of points on the wrong side of the line, and the margin of error for the line passing through all of them
# finally select the best based on this rating

# first find the peaks and troughs

def overall_line(idxs, vals):
    if len(idxs) <= 1:
        pm, zme = [np.nan, np.nan], [np.nan]
    else:
        p, r = np.polynomial.polynomial.Polynomial.fit(idxs, vals, 1, full=True)  # more numerically stable
        pm, zme = list(reversed(p.convert().coef)), r[0]
        if len(pm) == 1: pm.insert(0, 0.0)
    return pm

def calc_all(idxs, h, isMin, len_h, errpct, window, trendmethod, sortError):

    divide = list(reversed(range(len_h, -window, -window)))
    rem, divide[0] = window - len_h % window, 0
    if rem == window:
        rem = 0
    skey = 3 if sortError else 5

    min_h, max_h = min(h), max(h)
    scale = (max_h - min_h) / len_h
    fltpct = scale * errpct
    midxs = [[] for _ in range(len(divide) - 1)]
    for x in idxs:
        midxs[int((x + rem) / window)].append(x)
    mtrend = []
    for x in range(len(divide) - 1 - 1):
        m = midxs[x] + midxs[x + 1]
        mtrend.extend(trendmethod(m, h, fltpct, min_h, max_h))
    if len(divide) == 2:
        mtrend.extend(trendmethod(midxs[0], h, fltpct, min_h, max_h))
    mtrend = merge_lines(idxs, mtrend, h, fltpct)
    mtrend = [
        (pts, (res[0], res[1], res[2], res[3], res[4], measure_area((pts, res), isMin, h))) for pts, res in mtrend
    ]
    mtrend.sort(key=lambda val: val[1][skey])
    mwindows = window_results(mtrend, isMin, h, divide, window, skey)
    pm = overall_line(idxs, [h[x] for x in idxs])
    # print((pmin, pmax, zmne, zmxe))
    return pm, mtrend, mwindows