# Format data based on US holidays
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay
from pandas.tseries.offsets import CustomBusinessDay


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


def datefmt(xdate, cal=USTradingCalendar()):
    def mydate(x, pos):
        # print((x,pos))
        val = int(x + 0.5)
        if val < 0:
            return (xdate[0].to_pydatetime() - CustomBusinessDay(-val, calendar=cal)).strftime('%Y-%m-%d')
        elif val >= len(xdate):
            return (xdate[-1].to_pydatetime() + CustomBusinessDay(val - len(xdate) + 1, calendar=cal)).strftime(
                '%Y-%m-%d')
        else:
            return xdate[val].strftime('%Y-%m-%d')

    return mydate
