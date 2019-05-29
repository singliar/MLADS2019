import numpy as np
import pandas as pd

def is_column_sorted_ascending(df, time_colname):
    a = df[time_colname].values
    return np.all(a[:-1] <= a[1:])

def split_last_n_by_grain(df, n, time_column_name, grain_column_names, min_grain_length=1):
    """
    Group df by grain and split on last n rows for test, remaining first as train for each group.
    """
    
    paranoid = False
    
    gcols = grain_column_names + [time_column_name]
    if all([c in df.columns for c in gcols]):
        method = 'columns'
    elif set(df.index.names) == set(gcols):
        method = 'index'
    else:
        print('Your index levels are ' + ', '.join(df.index.names))
        print('Your dataframe key is ' + ', '.join(gcols))
        
        raise ValueError('time_column_name, grain_column_names must either both be in columns of df, ' + 
                          'or the index levels must be precisely time_column_name + grain_column_names');
    
    if method == 'columns' and not np.issubdtype(df[time_column_name].dtype, np.datetime64):
        print('WARNING: Your time column is not a datetime type, this function might split lexicographically')
    
    if method == 'index' and not np.issubdtype(df.head().index.get_level_values(time_column_name), np.datetime64):
        print('WARNING: Your time index is not a datetime type, this function might split lexicographically')
    
    if method == 'columns':        
        # group, then apply filter, which makes it a df again
        long_grains = df.groupby(grain_column_names, group_keys=False).filter(lambda g : len(g) >= min_grain_length)
        # now you have a flat df, group again        
        df_grouped = (long_grains.sort_values(time_column_name).groupby(grain_column_names, group_keys=False))         
        # flipping the order of group and sort would be natural but is hard in pandas. 
        # So sort first, group second, check third
        if paranoid:
            # this relies on stability of grouping, so assert occasionally
            assert(all([is_column_sorted(dfs, time_colname) for name, dfs in df_grouped]))
    elif method == 'index':
        long_grains = df.groupby(level=grain_column_names, group_keys=False).filter(lambda g : len(g) >= min_grain_length)
        df_grouped = (long_grains.sort_index().groupby(level=grain_column_names, group_keys=False))
        if paranoid:
            assert(all([dfs.index.is_monotonic_increasing for name, dfs in df_grouped]))
            
    df_train = df_grouped.apply(lambda dfg: dfg.iloc[:-n])    
    df_test = df_grouped.apply(lambda dfg: dfg.iloc[-n:])
    return df_train, df_test


def split_last_n_by_grain_tvt(df, n):
    """
    Group df by grain and split on last n rows for test, 
    next last n rows for val, remaining first few as train for each group.
    """
    if not df.issubdtype(df[time_column_name].dtype, np.datetime64):
        print('WARNING: Your time column is not a datetime type, this function might split lexicographically')
    
    df_grouped = (df.sort_values(time_column_name) # Sort by ascending time
                  .groupby(grain_column_names, group_keys=False))
    df_train = df_grouped.apply(lambda dfg: dfg.iloc[:-2*n])
    df_val = df_grouped.apply(lambda dfg: dfg.iloc[-2*n:-n])
    df_test = df_grouped.apply(lambda dfg: dfg.iloc[-n:])
    return df_train, df_val, df_test
    

def drop_grains_shorter_than(df, min_points, time_column_name, grain_column_names):
    """
    Removes grains that do not have at least min_points time points.
    
    For example, grains should be excluded that have fewer observations than
    the time horizon plus what is needed for cross-validation.
    The forecasts would be bad anyway!
    """
    df_grouped = df.sort_values(time_column_name).groupby(grain_column_names, group_keys=False)
    long_grains = df_grouped.filter(lambda g : len(g) >= min_points )
    return long_grains
    
def MPE(actual, pred):
    """
    Calculate mean percentage error ("bias").
    Remove NA and values where actual is close to zero
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    PE = 100*(actual_safe - pred_safe)/actual_safe
    return np.mean(PE)
    
    
def MAPE(actual, pred):
    """
    Calculate mean absolute percentage error.
    Remove NA and values where actual is close to zero
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    APE = 100*np.abs((actual_safe - pred_safe)/actual_safe)
    return np.mean(APE)


def SMAPE(actual, pred):
    """
    Calculate mean absolute percentage error.
    Remove NA and values where actual is close to zero
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    SAPE = 100*np.abs(2 * (actual_safe - pred_safe)/(actual_safe + pred_safe))
    return np.mean(SAPE)


def MSAPE(df_all, target_column_name, grain_column_names):
    """
    Calculate mean-scaled absolute percentage error.
    For a single series this is:
        \sum_h |F_h-A_h| / sum(|A_h|) * H
        
    In other words, the error is scaled by the mean absolute value
    of the series, not by the instantaneous F_h.
    """
    
    # FIXME: hardcoded values    
    df_err = df_all.copy()
    df_err['__automl_absactual'] = df_err[target_column_name].apply(math.fabs)
    df_err['__automl_abserror'] = (df_all[target_column_name] - df_all['predicted']).apply(math.fabs)        
    
    means = df_err[grain_column_names + ['__automl_absactual']].groupby(grain_column_names).mean()
    means.columns = ['__automl_mean_actual']
    
    errs = df_err[grain_column_names + ['__automl_abserror']].groupby(grain_column_names).mean()
    errs.columns = ['__automl_mean_abserror']
    
    df_metric = errs.merge(means, left_index=True, right_index=True)
    df_metric = df_metric[df_metric['__automl_mean_actual'] != 0 ]    
    df_metric['__automl_sape'] = 100 * df_metric['__automl_mean_abserror'] / df_metric['__automl_mean_actual']
    msape = np.mean(df_metric['__automl_sape'])    
    return msape, df_metric

# make a nice function that fills out the data frame with:
# * zeros for the target column
# * NaNs for the rest of the data
def fill_out_with_zeros(df, time_colname, grain_colnames, target_colname, freq = 'D', default_value=0):
    """
    For each series, fill in all zeroes from its first observation to last observation of all data.
    
    Expects all data to be present in the columns, with no index.
    * df                : pd.DataFrame - the input data frame
    * time_colname      : String       - name of the time column
    * grain_colnames    : List[string] - names of the grain columns
    * target_colname    : String       - name of the target (ts value) column
    * freq              : String       - frequency of rows to fill (pandas.Offset string like 'D', 'W')
    
    """
    # get the list of grains, with occurence count as a bonus
    grps = df.groupby(grain_colnames)
    unique_grains = grps.size().to_frame().rename(columns = {0 : "count"})    
    date_ranges = grps.agg({time_colname : ['min', 'max'] })
    date_ranges.columns = ['min','max']
    
    # make a dataframe of all dates for all grains with zero in target
    
    default_colname = '__automl_DefaultTarget'
    
    # don't this with a big join/filter because it might blow up too much
    indexes = []
    for index, row in unique_grains.iterrows():    
        grain_dates = pd.date_range(date_ranges.loc[index]["min"], date_ranges.loc[index]["max"], freq=freq).to_frame()
        for i, col in enumerate(grain_colnames):
            grain_dates[col] = index[i]        
        grain_dates.drop(columns=[0], inplace=True)
        grain_dates[default_colname] = default_value
        indexes.append(grain_dates)
        
    # put all the grain-level data frames together in one big df
    expected_values = pd.concat(indexes)    
    # the index is time, rename it to time_colname
    expected_values = expected_values.reset_index().rename(columns = {'index': time_colname})    
    # set same index for merge for the expected values and the indexed original df
    expected_values.set_index(grain_colnames + [time_colname], inplace=True)        
    indexed_df = df.set_index(grain_colnames + [time_colname])
    
    # merge on indexes
    merged_df = expected_values.merge(indexed_df, how='left', left_index=True, right_index=True)
    # replace those variables that are still null
    
    # most elegant wat to do a coalesce in pandas?
    # alternatively
    # merged_df[target_colname] = np.where(merged_df[target_colname].isnull(), df[default_colname], df[target_colname] )
    # merged_df.loc[merged_df[target_colname].isnull(), target_colname] = merged_df.loc[merged_df[target_colname].isnull(), default_colname]
    coalesced = merged_df[target_colname].combine_first(merged_df[default_colname])
    merged_df[target_colname] = coalesced
    
    return merged_df.drop(columns=default_colname)

#########################################################################
## Dropping grains

def drop_grains_from_dataset(X, y, grain_column_names, grains):
    
    # TODO: check that 1) grain_columns exist, 
    #                  2) grains have the same shape as the grain_columns
    #                  3) X and y are compatible shape
    
    # TODO: check that X_test is flat and not for example multiindexed with the grain
    X_ret = X.copy().set_index(grain_column_names)
    X_ret['__automl_target_column'] = y    
    X_ret.drop(grains, inplace=True)
    y_ret = X_ret.pop('__automl_target_column').values
    X_ret.reset_index(inplace=True)
    return X_ret, y_ret
    

def drop_untrained_grains_from_test(fitted_pipeline, X_test, y_test, grain_column_names):
    """
    Prints grains that are in the test set that the pipeline did not train for.
    Returns test dataset without the untrained grains.
    """
    test_grains = X_test[grain_column_names].drop_duplicates()
    
    untrained_grains = []
    for g in test_grains.itertuples(index=False):
        t = fitted_pipeline._ts_transformer.dict_latest_date.get(g)
        if t is None: 
            print("Will drop " + str(g))
            untrained_grains.append(tuple(g))
    
    return drop_grains_from_dataset(X_test, y_test, grain_column_names, untrained_grains)

#########################################################################
## Splitting datasets

def split_into_chunks_by_size(df, column, max_number_of_rows):
    """
    Split a dataframe into multiple data frames, each with max_number of rows
    
    Takes a dataframe, the column on whose value to split, and maximum 
    size of the resulting dataframes.
    
    Returns two aligned lists
    * list of data frames which partition the or
    * list of sets of `column` values
    """
    
    if not column in df.columns:
        raise ValueError('Splitting column must be in dataframe')
    
    sizes = df.groupby(column).size()
    
    # There may be a more efficient bin packing solver,
    # or if there are sufficiently many groups, one could simply
    # sample N/average_grain_size times without replacement
    # and hope to end up with fairly even groups by virtue of central limit
    from binpacking import to_constant_volume
    bins = to_constant_volume(sizes.to_dict(), max_number_of_rows)
    
    allframes = []
    allindices = []
    for idx, modelbin in enumerate(bins):
        minidf = df[ df[column].isin(set(modelbin.keys())) ]
        allframes.append(minidf)
        allindices.append(set(modelbin.keys()))
        
    return allframes, allindices

def split_into_chunks_by_groups(df, column, valuesets):
    """
    Split a dataframe into multiple dataframes.
    
    Take a list of sets of values of the `column`.
    
    For each valueset in the list, a dataframe will be made consisting of those
    rows which have the value of the `column` in the set.
    
    """
    
    if not column in df.columns:
        raise ValueError('Splitting column must be in dataframe')
        
    allframes = []
    for idx, valueset in enumerate(indices):
        minidf = df[ df[column].isin(valueset) ]
        allframes.append(minidf)
        
    return allframes


##############################
# Re making prediction context

# What what what did you just do?
# We filled in the entire y_query with nans, indicating they are 'question marks' to be forecasted.
# That happened to work because the prediction period immediately followed
# the training period, and recent values from training set were cached.
# But what if there was a "hole" - if I wanted to predict values starting 
# from some forecast origin later than the end of the training data?
# Then I would have to construct a prediction context. The context
# needs to contain the previous value of the target variable for each grain.
def make_forecasting_query(fulldata, forecast_origin, horizon, lookback):
    """
    This function will take the full dataset, and create the query
    to predict all values of the grain from the `forecast_origin`
    forward for the next `horizon` horizons. Context from previous
    `lookback` periods will be included.
    
    fulldata: pandas.DataFrame           a time series dataset. Needs to contain X and y.
    forecast_origin: datetime type       the last time we (pretend to) have target values 
    horizon: timedelta                   how far forward, in time units (not periods)
    lookback: timedelta                  how far back does the model look?
    
    Example:
    
    ```
    forecast_origin = pd.to_datetime('2012-09-01') + pd.DateOffset(days=5) # forecast 5 days after end of training
    print(forecast_origin)

    X_query, y_query = make_forecasting_query(data, 
                       forecast_origin = forecast_origin,
                       horizon = pd.DateOffset(days=7), # 7 days into the future
                       lookback = pd.DateOffset(days=1), # model has lag 1 period (day)
                      )
    ```
    
    """
    
    
    # TODO: this suffers from holes in data but let's let it slide for simplicity's sake
    # The right fix is to impute the missing values.
    X_past = fulldata[ (fulldata[ time_column_name ] > forecast_origin - lookback) &
                       (fulldata[ time_column_name ] <= forecast_origin)
                     ]
    X_future = fulldata[ (fulldata[ time_column_name ] > forecast_origin) &
                         (fulldata[ time_column_name ] <= forecast_origin + horizon)
                       ]
                      
    y_past = X_past.pop(target_column_name).values.astype(np.float)
    y_future = X_future.pop(target_column_name).values.astype(np.float)
    
    # Now take y_future and turn it into question marks
    y_query = y_future.copy().astype(np.float)  # because sometimes life hands you an int
    y_query.fill(np.NaN)
    
    print("X_past is " + str(X_past.shape) + " - shaped")
    print("X_future is " + str(X_future.shape) + " - shaped")
    print("y_past is " + str(y_past.shape) + " - shaped")
    print("y_query is " + str(y_query.shape) + " - shaped")
    
    X_pred = pd.concat([X_past, X_future])
    y_pred = np.concatenate([y_past, y_query])
    return X_pred, y_pred
    
