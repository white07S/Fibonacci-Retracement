import pandas as pd
import numpy as np
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