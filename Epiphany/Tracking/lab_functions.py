
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats

from scipy.odr import ODR, Model, Data, RealData

extension = 'C:\\Users\\abiga\\OneDrive - Durham University\\1. Third Year\\Advanced Labs\\1. Workable Folder\\Data\\'

'''
Functions to read and export excel / lvm data into voltage, time and monochromator arrays
respectively
'''

def data_excel(filename, startmono, endmono):
    

    df = pd.read_excel(filename)
    voltage = np.array(df['Untitled'])
    time = df['Time']
    
    time_secs = []
    
    for i in range(len(time)):
        time_secs.append((time[i] - time[0]).total_seconds())
        
    time_secs = np.array(time_secs)

    mono_steps = np.linspace(startmono, endmono, len(voltage))
        
    return voltage, time_secs, mono_steps

def data_lvm(file_location, startmono, endmono):
    
    ''' Input: lvm file location 
    '''
    ''' Output: plot of amplitude against time 
    in seconds and fractions of seconds as well as the full numpy arrays 
    of amplitude and time values '''

    ### ###

    ''' Importing lvm files from specified location 
        and
        Reading file using pandas and converting to numpy array '''

    data = np.array(pd.read_csv(file_location, skiprows = 12)) # skiprows = 12 gets to first 'channel' heading

    ## skip rows to get rid of header 
    # - when included an error pops up due to unmatching size of array

    ''' Extract voltages from lvm file using a for loop '''

    voltages = []

    for i in range(int(np.shape(data)[0] / 10)):
        voltages.append(float(data[9 + (10*i), 0])) ## rows 9, 19, 29 ... as long as 12 rows are skipped when reading the file

    voltages_numpy = np.array(voltages)

    ''' Extract timings of readings from lvm file using for loop '''

    time = []

    for j in range(int(np.shape(data)[0] / 10)):
        time.append(data[3 + (10*j), 0])     ## rows 3, 13, 23 .... as long as 12 rows are skipped when reading the file

    ''' converting time to pandas time delta type '''

    timedeltas = pd.TimedeltaIndex(time)

    ''' converting from H:M:S.f to total seconds and subtracting the zero time '''

    times_secs = timedeltas.total_seconds()
    times_shifted = np.array(times_secs - times_secs[0])

    #plt.scatter(times_shifted, voltages_numpy)
    #plt.show()

    mono_steps = np.linspace(startmono, endmono, len(voltages_numpy))

    return voltages_numpy, times_shifted, mono_steps

'''
Start / End Monochromator values and temperatures from file
'''

def read_info(column_name):

    data_info = pd.read_excel(extension + 'start_end_values.xlsx')

    info = np.array(data_info[column_name])
    startmono = info[0]
    endmono = info[1]
    starttemp = info[2]
    endtemp = info[3]

    return startmono, endmono, starttemp, endtemp

def chi_squared(model_params, model, x_data, y_data, y_error):
    return np.sum(((y_data - model(x_data, *model_params))/y_error)**2)

def chi_squared_odr(model_params, model, x_data, y_data, y_error):
    return np.sum(((y_data - model(model_params, x_data))/y_error)**2)

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

    p_value = scipy.stats.chi2.sf(chisq_min, dof)

    yfit = function(xdata, *parameters)

    return yfit, parameters, perrors, chisq_reduced, p_value, dof

def fit_labs_odr(xdata,ydata, xerrs, yerrs, function_vect, initial_guess):

    dof = np.size(xdata) - np.size(initial_guess) ## degrees of freedom

    if np.size(yerrs) == 1:

        yerr_extended = np.zeros(np.size(xdata))
        for i in range(np.size(yerr_extended)):
            yerr_extended[i] = yerrs

        # Create Model with two guasses
        modelODR = Model(function_vect)
        # Create Data
        mydata = RealData(xdata, ydata, sx=xerrs, sy=yerr_extended)
        # Set up ODR
        myodr = ODR(mydata, modelODR, beta0=initial_guess)
        # Run Fit and Examine Output
        myoutput = myodr.run()
        #myoutput.pprint()
        # Extract Info
        ODRparams = myoutput.beta
        ODRparams_err = myoutput.sd_beta

    else:
    
        # Create Model with two guasses
        modelODR = Model(function_vect)
        # Create Data
        mydata = RealData(xdata, ydata, sx=xerrs, sy=yerrs)
        # Set up ODR
        myodr = ODR(mydata, modelODR, beta0=initial_guess)
        # Run Fit and Examine Output
        myoutput = myodr.run()
        #myoutput.pprint()
        # Extract Info
        ODRparams = myoutput.beta
        ODRparams_err = myoutput.sd_beta

    chisq_min = chi_squared_odr(ODRparams,
                        function_vect, 
                        xdata, 
                        ydata, 
                        yerrs)
    
    chisq_reduced = chisq_min / dof

    p_value = scipy.stats.chi2.sf(chisq_min, dof)

    yfit = function_vect(ODRparams, xdata)

    return yfit, ODRparams, ODRparams_err, chisq_reduced, p_value, dof

def norm_residuals(x,y,yerr,model, params):

    diff = y - model(x, *params)

    return diff / yerr

def norm_residuals_odr(x,y,yerr,model, params):

    diff = y - model(params, x)

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

