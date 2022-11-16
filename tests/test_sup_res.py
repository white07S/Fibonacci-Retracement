from src.codeReview.sup_res.sup_res import SupResFinder
from src.codeReview.sup_res.extrema.extrema import *


def test_SupResFinder():
    data = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0]
    data = [float(x) for x in data]
    result = (
        (
            [6, 12, 18], [0.0, 0.0], [([6, 12, 18], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))],
            [[([6, 12, 18], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))]]
        ),
        (
            [3, 9, 15, 21], [0.0, 3.0], [([3, 9, 15, 21], (0.0, 3.0, 0.0, 0.0, 0.0, 0.0))],
            [[([3, 9, 15, 21], (0.0, 3.0, 0.0, 0.0, 0.0, 0.0))]]
        )
    )
    spf = SupResFinder()
    assert result == spf.calc_support_resistance(data, extmethod=METHOD_NAIVE)
    assert result == spf.calc_support_resistance(data, extmethod=METHOD_NAIVECONSEC)
    # TODO - what about the other method? Merge with what's in main?
    # assert result == calc_support_resistance(data)


if __name__ == '__main__':
    # TODO - run tests locally - let me know if it interferes with sth else
    test_SupResFinder()
