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


'''A function to tidy up the raw gaussian data. Implements measures to normalise and transform data such that
the only fitting parameter becomes the waist of the beam, W, AND THE POSITION OF THE CENTRE, C '''

def data_trim(distances_, amplitudes_, cutoff):

    '''A function to tidy up the raw gaussian data. Implements measures to normalise and transform data such that
    the only fitting parameter becomes the waist of the beam, W, AND THE POSITION OF THE CENTRE, C '''

    new_distances = []
    new_amps = [] # some empty lists to append to at the end of the loop

    for j in range(len(distances_)): # iterate through distances (amps)

        # Get rid of vertical offset - to reduce a fitting parameter
        subt_amp = amplitudes_[j] - np.min(amplitudes_[j])

        # Normalise data by its maximum value
        norm_amp = subt_amp / np.max(subt_amp)

        dist_max = 0 # to add to later

        ''' NB: Do not translate data to zero peak as this was causing issues with fitting'''

        # # Move data so centred on zero - reduce fitting params further
        # for i in range(norm_amp.size):
        #     if norm_amp[i] == np.max(norm_amp):
        #         dist_max = distances_[j][i]

        # shifted_distances = distances_[j] - dist_max

        shifted_distances = distances_[j]

        cut_dist = 0
        cut_amps = 0

        if len(shifted_distances) > cutoff:
            cut_dist = shifted_distances[0:cutoff]
            cut_amps = norm_amp[0:cutoff]

            # Add adjusted data set to the new large lists
            new_distances.append(cut_dist)
            new_amps.append(cut_amps)

        else:
            new_distances.append(shifted_distances)
            new_amps.append(norm_amp)

    return new_distances, new_amps

### FITTING FUNCTIONS

def chi_squared(model_params, model, x_data, y_data, y_error):
    return np.sum(((y_data - model(x_data, *model_params))/y_error)**2)

def fit_labs(xdata,ydata, yerrors, function, initial_guess):

    dof = np.size(xdata) - np.size(initial_guess) ## degrees of freedom

    if np.size(yerrors) == 1:

        yerr_extended = np.zeros(np.size(xdata))
        for i in range(np.size(yerr_extended)):
            yerr_extended[i] = yerrors

        parameters, covariance = curve_fit(function, xdata, ydata, sigma = yerr_extended, absolute_sigma = True, p0 = initial_guess, maxfev = 50000)

    else:
    
        parameters, covariance = curve_fit(function, xdata, ydata, sigma = yerrors, absolute_sigma = True, p0 = initial_guess, maxfev = 50000)

    perrors = np.sqrt(np.diag(covariance))

    chisq_min = chi_squared(parameters,
                        function, 
                        xdata, 
                        ydata, 
                        yerrors) 
    
    chisq_reduced = chisq_min / dof

    # p_value = scipy.stats.chi2.sf(chisq_min, dof)

    yfit = function(xdata, *parameters)

    return yfit, parameters, perrors, chisq_reduced

''' Gaussian with two parameters'''

def Gauss(x, W, C): ### Defien Gaussian with TWO parameters
    y = np.exp((-2*((x-C)**2) / W**2))
    return y

''' Define w vs z theoretical relation -
we use THREE fitting parameters,
w0,z0 and Rayleigh Range (RR)'''
'''... Theory for Gaussian beam says that w0 and RR are related but fit doesnt work with only 2 parameters ...'''

def WvsZ(x, min_w,C,RR):
    # define rayleigh range
    #RR = (np.pi * min_w**2) / wavelength
    return min_w * np.sqrt(1 + ((x-C)/RR)**2)

wavelength_ = 635e-9

def WvsZ_2(x, min_w,C):
    # define rayleigh range
    RR = (np.pi * min_w**2) / (wavelength_)
    return min_w * np.sqrt(1 + ((x-C)/RR)**2)

### RAY OPTICS MODEL FUNCTIONS

''' Getting wavenumber, k, from wavelength, wl, input in microns, OUTPUT in per metre'''

def find_k(wl): # in microns
    return 2*np.pi / (wl*10**(-6)) # wavenumber, SI units

''' Getting V number from numerical aperture, wavenumber and core radius'''

def V_number(num_ap, waveno, radius):
    return num_ap*waveno*radius

''' Defining the Marcuse Relation below, to compute theoretical minimum waist from V number and core radius '''

# INPUT: SI units
# OUTPUT: microns

def marcuse(V,radius):
    return radius*( 0.65 + 1.619*V**(-3/2) + 2.879*V**(-6) )*10**6

''' Calculating the NA of the tapered fibre '''

def NA_taper(clad_d, new_d, NA_):
    return (new_d/clad_d) * NA_

def new_a(a_, NA_, NA_taper):
    return (NA_taper/NA_) * a_

''' A function to check that the independant rayleigh range and min waist are / are not producing the wavelength expected
This relation only holds for a perfectly gaussian beam ...'''

def wavelength_check(waist,RR): # input - microns
    return ((np.pi*(waist)**2) / RR) * 10**(3) #s returns in nanometers!

''' Residual and histogram functions '''

def norm_residuals(x,y,yerr,model, params):

    diff = y - model(x, *params)

    return diff / yerr

def functional_approach(x,func,alphax):

    return np.abs(func(x + alphax) - func(x))

def functional_approach_many(variables, func, uncertainties):

    sum_ = []

    for i in range(np.size(variables)):

        sum_.append( np.abs( func(variables[i] + uncertainties[i]) - func(variables[i]) ) ** 2)

    return np.sqrt(np.sum(sum_))

def histogram_plot(residuals):

    sorted_res = sorted(residuals)

    mean = np.mean(sorted_res)
    st_dev = np.std(sorted_res)

    normal_dist = scipy.stats.norm.pdf(sorted_res, mean, st_dev)

    return mean, st_dev, sorted_res, normal_dist
