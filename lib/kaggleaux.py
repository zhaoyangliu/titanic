# Kaggel Auxillary Functions
# Copy-Write AGC 2013

def cross_validate_df(df, percent):
    '''
    Randomly samples a percentage of a pandas dataframe for cross validation or for down sampleing
    
    ins
    --
    df -- to be sampled **expects an ordinal index ** ie. 0 - 100
    percent -- as an int of the percentage to be sampled (ie 25 for 25%)
    
    out
    --
    (df_h1, df_h2) -- both halfs of the split data frame
    
    ex.
    small_df_half,large_df_half  = cross_validate_df(df,33)
    '''
    sample_percentage = int(np.round(((percent * df.index.size) / float(100))))
    rows = np.random.randint(0, df.index.size, sample_percentage)
    df_h1 = df.ix[rows]
    df_h2 = df.drop(rows)
    return (df_h1, df_h2)

def dataframe_kfolds(df):
    '''
    standard kfolds cross cross_validate

    in 
    --
    DataFrame

    out
    --
    dataframe 10%, dataframe 90% -- random splits using python's random.choice()

    *suitable for large datasets
    '''
    import random
    import numpy as np
    obs_to_extract = df.index
    
    # get random 10% subsection of data
    for i in xrange(0, round(df.index.size * 0.10)):
        ten_pct_sample.append(random.choice(obs_to_extract))
    
    ninety_pct_sample = np.delete(obs_to_extract, ten_pct_sample) # gives us the complement
    
    return df.ix[ten_pct_sample,:], df.ix[ninety_pct_sample,:]
    

def dataframe_welch_ttest(df, desc_frame):
    '''
    Takes in a dataframe, and a described dataframe ( from the desribe_frame() method)
    
    returns
    
    t-statistic and p-value for each feature in a pandas dataframe
    ''' 
    import scipy as sp
    import numpy as np
    desc_frame['t-statistic'] = np.nan
    desc_frame['p-value'] = np.nan
    
    for name,item in df.iteritems():
        try :
            res = sp.stats.ttest_ind(df[name][df.sued_bool == 0].dropna(), 
                                    df[name][df.sued_bool == 1].dropna(),
                                    equal_var=False
                                    )
        except:
            res = (np.nan, np.nan)
            
        desc_frame.ix[name,'t-statistic'], desc_frame.ix[name,'p-value'] = res
            
def cat_bool_maker(s):
    '''
    when used with the pandas df.series.apply() method,  it will create a boolean cat var. 
    if values exist the bool will register as 1 
    if nan vaues esit the bool will register as 0

    '''
    if np.isnan(s) == True:
        return 0
    else:
        return 1
    
def columns_to_str(column_list, return_list=0):
    '''
    takes the pandas df. columns method and returns a list with items as strings for easy implementaiton into pandas fucntions
    
    ins
    --
    column_list -- ussually from pandas df.columns function but can be any list
    
    return_list -- optional param to return the str list ( use 1 to return)
    
    out 
    --
    a list with str elements
    
    ex.
    
    columns_to_str(df.columns)
    
    Changes
    
    Index([x3yv_E, x3yv_D, x1yv_E, x1yv_D], dtype=object)
    
    to
    
    Index(['x3yv_E', 'x3yv_D', 'x1yv_E', 'x1yv_D'], dtype=object)
    '''
    if return_list == 1:
        output = []
        for i in column_list:
            output.append(str(i))
        return output
    
    print "[%s]" % "', '".join(map(str, column_list))

def columns_to_plus(column_list, return_list=0):
    '''
    takes the pandas df. columns method and returns a list with items connected with a plus for easy implementaiton into patys's dmatrices function
    
    ins
    --
    column_list -- ussually from pandas df.columns function but can be any list
    
    return_list -- optional param to return the str list ( use 1 to return)
    
    out 
    --
    a list with elements conncted with a plus 
    
    ex.
    
    columns_to_str(df.columns)
    
    Changes
    
    Index([x3yv_E, x3yv_D, x1yv_E, x1yv_D], dtype=object)
    
    to
    
    Index([x3yv_E + x3yv_D + x1yv_E + x1yv_D], dtype=object)
    '''
    if return_list == 1:
        output = []
        for i in column_list:
            output.append(str(i))
        return output
    
    print "[%s]" % " + ".join(map(str, column_list))

def add_to_model_subspace(subspace_def, var):
    '''
    come on, it does what it says.
    
    basic check to easily add vars to model subspace definitions. 
    
    ins
    --
    var - list, that thing(s) your trying to add
    subspace_def - list defining your models subspace
    
    out
    --
    subspace_def - list defining your models subspace w/ var added
    '''
    for i in range(0, len(var)):  
        if (var[i] in subspace_def) == False:
            var[i] = str(var[i])
            assert isinstance(var[i], str)
            subspace_def.append(var[i])
    
    #return subspace_def
    
def ml_formula(y, df):
    '''
    a simple function to create a formula using all available features for patsy dmatrices function. 

    ins  
    --
    y = a string of the variable your trying to predict

    df = the data frame your trying to make your predctions on 

    outs 
    ---
    a string of the desired formula
    '''
    formula = y +' ~ '
    for i, val in enumerate(df.columns):
        if i == 0 and val != y:
            formula += val
        if i != 0 and val != y:
            formula += ' + ' + val
    return formula 



def progress(i, num_tasks):
    '''
    A simple textual progress bar
    
    Ins
    --
    i = should be an iterable value to measure the progress of your taks
    num_tasks = total number of tasks

    outs 
    --
    A progress bar like [#########  ]
    '''
    import sys

    progress = "\r["
    
    for _ in range (0,i):
        progress += "#"
    
    for _ in range (i, num_tasks):
        progress += " "
    
    progress += "]"
    
    sys.stdout.write(progress)
    sys.stdout.flush()
    
def cat_clean(s):
    """
    Cleans categorical variables of NaNs by setting the NaN to 0 so it can be processed by a patsy dmatrices function.

    Inputs
    --
    s = cat value

    output
    --
    cat value or int 0 in place of a NaN val
    """
    import numpy as np
    
    if isinstance(s, str) == False and np.isnan(s) == True:
        s = 0
    return s 

def predict(test_data, results, i):
    """ 
    Returns a NumPy array of independent variable predictions of a test file based on your regression of a train file. Built for speed
    
    Parameters
    --
    Test_data: should be test data you are trying to predict in a pandas dataframe 
    results: should be dict of your models results wrapper and the formula used to produce it. 
        ie.  
        results['Model_Name'] = {[<statsmodels.regression.linear_model.RegressionResultsWrapper> , "Price ~ I(Supply, Demand)] }
    i: should be the name of your model. You can iterate through the results dict. 
    --
   
    Returns
    --
    Predictions in a flat NumPy array. 
    AGC 2013
    """
    import numpy as np
    from pandas import DataFrame
    from patsy import dmatrices

    
    model_params = DataFrame(results[i][0].params)
    formula = results[i][1]
    
    # Create reg friendly test dataframe
    yt, xt = dmatrices(formula, data=test_data, return_type='dataframe')

    
    # remove extraneous features for efficiency 
    to_drop = list()
    to_drop[:] = [] # Empty list, in case cells are executed out of order
    for c in xt.columns:
        if c not in model_params.index:
            to_drop.append(c)
    xt = xt.drop(to_drop, axis=1)
    
    to_drop[:] = [] # Empty list
    for c in model_params.index : 
        if c not in xt.columns:
            to_drop.append(c)
    model_params = model_params.drop(to_drop)
    
    # Convert to NumPy arrays for performance
    model_params = np.asarray(model_params)
    yt = np.asarray(yt)
    yt = yt.ravel()
    xt = np.asarray(xt)

    
    # Use our models to create predictions
    row, col = xt.shape
    model_params = model_params.ravel() # flatten array
    model_array = []
    
    for _ in xrange(row):
            model_array.append(model_params)
    model_array = np.asarray(model_array)
    
    # Multiply matrix together 
    predictions = np.multiply(xt, model_array)
    predictions = np.sum(predictions, axis = 1)

    return predictions

def quater_maker(d):
    """
    Pareses dates and returns the appropriate quarter.
    --
    Parameters
    d: a python date time object
    --
    Returns:
    The quarter in a string
    
    AGC 2013
    """
    r = ''
    Q1 = [1, 2, 3]
    Q2 = [4, 5, 6]
    Q3 = [7, 8, 9]
    Q4 = [10, 11, 12]
    if d.month in Q1:
        r = 'Q1'
    if d.month in Q2:
        r = 'Q2'
    if d.month in Q3:
        r = 'Q3'
    if d.month in Q4:
        r = 'Q4'
    return r
        
      #if date is within daterange, Q1 ect. then output ex: 2011 Q1
      #manipulate months. say if  0<m4: return YYYY + Q1 ect.
      # then apply to dateline

def score_rmsle(y, df, df2, p = 0):
    '''
    ins 
    --
    y =  what your trying to predict. must be a string. ie. 'SalesPrice'
    df =  your predictions
    df2 = the solutions set
    p = option to print rmsle as string; 0 = dont print, 1 = print 
    outs
    --
    prints rmsle
    rmsle as a float
    '''
    import numpy as np
    
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsmle = np.sqrt(np.mean(np.power(np.log(actual + 1) - np.log(prediction + 1), 2)))
    
    if p == 1 :
        print "rsmle: " + str(rsmle)
    return rsmle

def score_rmse(y, df, df2, p = 0 ):
    """
    ins 
    --
    y =  what your trying to predict. must be a string. ie. 'SalesPrice'
    df =  your predictions
    df2 = the solutions set
    p = option to print rmsle as string; 0 = don't print, 1 = print

    outs
    --
    prints rmse
    rmse as a float
    """
    import numpy as np

    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])
   
    rsme = np.sqrt(np.mean(np.power((actual - prediction), 2)))
    
    if p == 1:
        print "rsme: " + str(rsme)
    
    return rsme

def unwanted_pals(x, s = .1):
    '''
    Inputs 
    --
    x should be a pandas series of the results.pvalues of your model
    s = significance level, default at 90% confidence
    
    Outs
    --
    returns a list of columns below significance level
    '''
    dropl = list()
    for i,z in enumerate(x):
        if z>s:
             dropl.append(x.index[i])
    return dropl

def stock_price_at_date(x, ticker, lag=0):
    '''
    ins
    --
    x should be the date your looking for 
    ticker should be the stock ticker
    lag should be # of days to lag stock price

    outs
    --
    stock price.
    '''
    from pandas.io.data import DataReader
    from datetime import date, timedelta
 
    x = (x - timedelta(days = lag))
    r = DataReader(ticker,start=x, end=x, data_source='yahoo')
    r = r.ix[0,5]
    return r

def side_by_side(*objs, **kwds):
    '''
    created by wes mickinney, it only exists here becuase I use this function all the time. 
    '''
    from pandas.core.common import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print adjoin(space, *reprs)

def describe_frame(df):
    """
    ins 
    ---
    df = dataframe you want to describe

    outs
    ---
    descriptive stats on your dataframe, in dataframe.

    agc2013
    """
    from pandas import Series, DataFrame
    sum_stats = []
    for i in df.columns:
        x = Series(df[i].describe())
        x.name = i
        sum_stats.append(x)
    stats = DataFrame(sum_stats)
    stats
    return stats

def bin_residuals(resid, var, bins):
    '''
    Compute average residuals within bins of a variable.
    
    Returns a dataframe indexed by the bins, with the bin midpoint,
    the residual average within the bin, and the confidence interval 
    bounds.
    
    ins 
    -- 
    resid, var, bins

    out
    --
    bin DataFrame

    '''
    from pandas import DataFrame, qcut
    import NumPy as np
    # use scipy's binned stat method
    resid_df = DataFrame({'var': var, 'resid': resid})
    resid_df['bins'] = qcut(var, bins)
    bin_group = resid_df.groupby('bins')
    bin_df = bin_group['var', 'resid'].mean()
    bin_df['count'] = bin_group['resid'].count()
    bin_df['lower_ci'] = -2 * (bin_group['resid'].std() / 
                               np.sqrt(bin_group['resid'].count()))
    bin_df['upper_ci'] =  2 * (bin_group['resid'].std() / 
                               np.sqrt(bin_df['count']))
    bin_df = bin_df.sort('var')
    return(bin_df)

def plot_binned_residuals(bin_df):
    '''
    Plotted binned residual averages and confidence intervals.
    
    ins
    --
    bin_df ie from bin_residuals(resid, var, bins)
    outs
    --
    pretty plots
    '''
    import matplotlib as plt

    plt.plot(bin_df['var'], bin_df['resid'], '.')
    plt.plot(bin_df['var'], bin_df['lower_ci'], '-r')
    plt.plot(bin_df['var'], bin_df['upper_ci'], '-r')
    plt.axhline(0, color = 'gray', lw = .5)
