import pandas as pd
import numpy as np
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
        USLaborDay, USThanksgivingDay
from pandas.tseries.offsets import CustomBusinessDay
def check_num_alike(h):
    if type(h) is list and all([isinstance(x, (bool, int, float)) for x in h]): 
        return True
    elif type(h) is np.ndarray and h.ndim==1 and h.dtype.kind in 'biuf': 
        return True
    else:
        if type(h) is pd.Series and h.dtype.kind in 'biuf': 
            return True
        else: 
            return False

def datefmt(xdate, cal=None):
    class USTradingCalendar(AbstractHolidayCalendar):
        rules = [
            Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday('Christmas', month=12, day=25, observance=nearest_workday)
        ]
    if cal == None: cal = USTradingCalendar()
    def mydate(x,pos):
        #print((x,pos))
        val = int(x + 0.5)
        if val < 0: return (xdate[0].to_pydatetime() - CustomBusinessDay(-val, calendar=cal)).strftime('%Y-%m-%d')
        elif val >= len(xdate): return (xdate[-1].to_pydatetime() + CustomBusinessDay(val - len(xdate) + 1, calendar=cal)).strftime('%Y-%m-%d')
        else: return xdate[val].strftime('%Y-%m-%d')
    return mydate