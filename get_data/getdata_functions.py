'''
File with functions to get the data and save it as csv file
'''

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
import config


def get_data(API):
    '''
    Use an API to request data from a homepage and save it as json. 

    Parameter
    --------
    API: str
        The API for the http request.

    Return
    --------
    data: json
        json file of the responded data.
    '''
    response = requests.get(API,
            auth = HTTPBasicAuth(config.user, config.key))
    print(response)
    data = response.json()

    return data




def make_df(json_input, csv_output):
    '''
    Function to create a pd.DataFrame form a json-file and save the DataFrame as csv-file.

    Parameter
    --------
    json_input: str
        The json file.
    csv_ouput: str
        Filename for the csv-file.

    Return
    --------
    df: pd:DataFrame
        Created Dataframe out of the json.file.
    '''
    
    lst_keys = []
    for key in json_input[0].keys():
        lst_keys.append(key)

    dict_keys = {}
    for key in lst_keys:
        dict_keys[key] = ''

    lst_values = []
    for number in range(0,len(json_input)):
        for value in json_input[number].values():
            lst_values.append(value)

    #creare list sorted by values 
    lst_values_sort = []
    for number in range(0,len(lst_keys)):
        for n in np.arange(number, len(lst_values), len(lst_keys)):
            lst_values_sort.append(lst_values[n])

    #create lists and use them as values for keys in dict
    for number in range(0,16):
        lst_lst_values = []
        for range_begin in np.arange(0,len(lst_values_sort),len(np.arange(0, len(lst_values_sort), len(lst_keys)))):
            range_end = range_begin +len(np.arange(0, len(lst_values_sort), len(lst_keys)))
            lst_lst_values.append(lst_values_sort[range_begin :range_end])

    for number in range(0,len(dict_keys)):
        dict_keys[list(dict_keys.keys())[number]] = lst_lst_values[number]

    df = pd.DataFrame(dict_keys)

    df.to_csv(csv_output)
    return df

