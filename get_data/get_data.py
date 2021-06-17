'''
File to create and save csv files
'''

import pandas as pd
import numpy as np
from getdata_functions import get_data, make_df



data = get_data('API') #use API to get Data, copy key in config.py file

make_df(data,'../data/data.csv')
