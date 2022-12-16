from src.models.train_model import train
from src.codeReview.sup_res_preliminary import calc_support_resistance, METHOD_NAIVE, METHOD_NAIVECONSEC

def test():
    assert train() == "Hello World"

def test_calc_test():
    data = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0]
    data = [float(x) for x in data]
    result = (([6, 12, 18], [0.0, 0.0], [([6, 12, 18], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))], [[([6, 12, 18], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))]]), ([3, 9, 15, 21], [0.0, 3.0], [([3, 9, 15, 21], (0.0, 3.0, 0.0, 0.0, 0.0, 0.0))], [[([3, 9, 15, 21], (0.0, 3.0, 0.0, 0.0, 0.0, 0.0))]]))
    assert result == calc_support_resistance(data, extmethod=METHOD_NAIVE,accuracy=8)