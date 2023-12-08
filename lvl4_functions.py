''' Importing necessary functions'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from os import listdir
import scipy

''' A function which scans for csv files in the global path given and imports the raw data as a set of arrays within a large list
NB: Does not import the last csv file!! this is the zs file which is used later'''

def import_data(global_path):

    # Function which finds filenames of csvs in a folder
    def find_csv_filenames( path_to_dir, suffix=".csv" ):
        filenames = listdir(path_to_dir)
        return [ filename for filename in filenames if filename.endswith( suffix ) ]
    
    def sort_int(examp):
        pos = 1
        while examp[:pos].isdigit():
            pos += 1
        return examp[:pos-1] if pos > 1 else examp

    # the list of files in the folder
    files_list = find_csv_filenames(global_path)
    
    sorted(files_list, key=sort_int) # sort files

    files_list = files_list[:-1] # -1 to not include 'zs' csv file - import this later

    # empty lists to put values in
    distances = []
    amps = []

    print(files_list)

    # loop through file names in directory
    for f in range(len(files_list)):

        # Import an execl sheet as dataframe, called 'Values1'
        # NB: Image J seems to have saved this 'excel sheet' as a csv file
        df_test = pd.read_csv(global_path + files_list[f])

        # Extracing a column by title and converting data to array
        distances.append(np.array(df_test['Distance_(microns)']))
        amps.append(np.array(df_test['Gray_Value']))

    # returning arrays within two big lists
    return distances, amps