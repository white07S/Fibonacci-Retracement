from src.codeReview.utils import get_bestfit, get_bestfit3


def get_trend(Idxs, h, fltpct, *args, **kwargs):  # kwargs are not used (added for the factory patter to work)
    trend = []
    # unfortunately an O(n(n-1)(n-2))=O((n^2-n)(n-2))=O(n^3-3n^2-2n)~=O(n^3) algorithm
    # but meets the strict definition of a trendline
    for x in range(len(Idxs)):
        for y in range(x + 1, len(Idxs)):
            # slope = (h[Idxs[x]] - h[Idxs[y]]) / (Idxs[x] - Idxs[y])
            # m=dy/dx #if slope 0 then intercept does not exist constant y where y=b
            # intercept = h[Idxs[x]] - slope * Idxs[x] #y=mx+b, b=y-mx
            for z in range(y + 1, len(Idxs)):
                # distance = abs(slope * Idxs[z] + intercept - h[Idxs[z]])
                # #distance to y value based on x with slope-intercept
                trend.append((
                    [Idxs[x], Idxs[y], Idxs[z]],
                    get_bestfit3(Idxs[x], h[Idxs[x]], Idxs[y], h[Idxs[y]], Idxs[z], h[Idxs[z]])
                ))

    return list(filter(lambda val: val[1][3] <= fltpct, trend))


def get_trend_opt(Idxs, h, fltpct, *args, **kwargs):
    slopes, trend = [], []
    # O(n^2*log n) algorithm
    for x in range(len(Idxs)):
        slopes.append([])
        for y in range(x + 1, len(Idxs)):
            slope = (h[Idxs[x]] - h[Idxs[y]]) / (
                    Idxs[x] - Idxs[y])  # m=dy/dx #if slope 0 then intercept does not exist constant y where y=b
            # intercept = h[Idxs[x]] - slope * Idxs[x] #y=mx+b, b=y-mx
            slopes[x].append((slope, y))
    for x in range(len(Idxs)):
        slopes[x].sort()  # key=lambda val: val[0])
        CurIdxs = [Idxs[x]]
        for y in range(0, len(slopes[x])):
            # distance = abs(slopes[x][y][2] * slopes[x][y+1][1] + slopes[x][y][3] - h[slopes[x][y+1][1]])
            CurIdxs.append(Idxs[slopes[x][y][1]])
            if len(CurIdxs) < 3:
                continue
            res = get_bestfit([(p, h[p]) for p in CurIdxs])
            if res[3] <= fltpct:
                CurIdxs.sort()
                if len(CurIdxs) == 3:
                    trend.append((CurIdxs, res))
                    CurIdxs = list(CurIdxs)
                else:
                    CurIdxs, trend[-1] = list(CurIdxs), (CurIdxs, res)
                # if len(CurIdxs) >= MaxPts: CurIdxs = [CurIdxs[0], CurIdxs[-1]]
            else:
                CurIdxs = [CurIdxs[0], CurIdxs[-1]]  # restart search from this point

    return trend
