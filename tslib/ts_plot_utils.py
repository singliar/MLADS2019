# Time series plot utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(df, grain_col, grain_val, 
                     value_col, time_col, transform):     
    xaxis = df[df[grain_col] == grain_val][time_col]
    yline = df[df[grain_col] == grain_val][value_col].apply(transform)
    plt.plot(xaxis, yline, label=grain_val)    
    
def plot_all_series(df, 
                    value_col, grain_col, time_col, 
                    transform = lambda x:x,
                    legend = True):     
    
    for loc in pd.unique(df[grain_col]):
        plot_time_series(df, grain_col, loc, value_col, time_col, transform)
    
    smin = min(df[value_col].apply(transform)) * 1.05
    smax = max(df[value_col].apply(transform)) * 1.05
    plt.ylim([smin, smax])
    plt.xticks(rotation=60)
    plt.ylabel(value_col)
    if legend:
        plt.legend()
    plt.show()
    
    
def hor_bar_chart(importances, features,n):
    """
    plot the importances of the top n features as a horizontal bar chart
    
    Input:
        importances: list<float> - the importance values, sorted descending
        
        features: list<string> feature names, aligned to importances
        
        n: take the first n features. importances and 
    """

    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(n)
    ax.barh(y_pos, importances[0:n], align='center',color='green')
    # ax.yaxis.tick_right()    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features[0:n])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.tick_params(axis='y', direction='in')
    ax.set_xlabel('Importance')
    ax.set_title('Predictive Importance of Exceptions')
    plt.show()
    
# plot the time series of price vs quantity
def quantity_price_plot(t,q,p,a):
    """
    Takes time, quantity, and price series.
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Quantity", color=color, fontsize=24)
    ax1.plot(t, q, color=color)   # step would be more correct but harder to read
    ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
    
    ax1.fill_between(t.values, 
                     np.zeros_like(t, dtype=np.float), 
                     max(q) * np.ones_like(t, dtype=np.float) * a, 
                     facecolor='blue', alpha=0.2, 
                     step='mid' # 'pre' would be more correct but harder to read
                    )

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel("Price", color=color, fontsize=24)  # we already handled the x-label with ax1
    ax2.plot(t, p, color=color)   # step would be more correct but harder to read
    ax2.tick_params(axis='y', labelcolor=color, labelsize=18)

    fig.tight_layout()  
    plt.show()

def oj_q_p_plot(df, store, brand, since):
    
    time_col = "WeekStarting"    
    quantity_col = "Quantity"
    price_col = "Price"
    advert_col = "Advert"
    
    t = df[(df['Store'] == store) & (df['Brand'] == brand) & (df[time_col] >=since)][time_col]
    q = df[(df['Store'] == store) & (df['Brand'] == brand) & (df[time_col] >=since)][quantity_col]
    p = df[(df['Store'] == store) & (df['Brand'] == brand) & (df[time_col] >=since)][price_col]
    a = df[(df['Store'] == store) & (df['Brand'] == brand) & (df[time_col] >=since)][advert_col]
    quantity_price_plot(t,q,p,a)


def plot_forecast(X_trainval, y_trainval,
                  X_test, y_test, y_pred,
                  target_column_name,
                  time_column_name,
                  actual_color='blue',
                  pred_color='green',
                  filter_dict = None):
    """
    Plot a forecast.
    
    Params:
        filter_dict: a mapping from column to list of values
                     Will be used to filter data down. Only rows
                     that contain the listed values for column will be retained.                     
    """
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    test_set = X_test.copy()
    test_set[target_column_name] = y_test
    test_set['prediction'] = y_pred
    
    train_set = X_trainval.copy()
    train_set[target_column_name] = y_trainval
    train_set['prediction_lower95'] = y_trainval
    train_set['prediction_upper95'] = y_trainval
    train_set['prediction_lower50'] = y_trainval
    train_set['prediction_upper50'] = y_trainval

    if filter_dict is not None:
        for col, values in filter_dict.items():
            if not isinstance(values, list):
                values = [ values ]            
            train_set = train_set[ train_set[col].isin(values)]
            test_set = test_set[ test_set[col].isin(values)]            
    
    rmse = np.sqrt(mean_squared_error(test_set[target_column_name], test_set['prediction']))
    test_set['prediction_lower95'] = test_set['prediction'] - 1.96 * rmse
    test_set['prediction_upper95'] = test_set['prediction'] + 1.96 * rmse
    test_set['prediction_lower50'] = test_set['prediction'] - 0.67 * rmse
    test_set['prediction_upper50'] = test_set['prediction'] + 0.67 * rmse
    
    train_set.sort_values(time_column_name, inplace=True)
    test_set.sort_values(time_column_name, inplace=True)
    
    # filter the train set to the same length as test set to avoid squished plot    
    _ , train_set = split_last_n_by_grain(train_set, 2*len(test_set))
    
    plt.plot(train_set[time_column_name], train_set[target_column_name], c=actual_color)   # train-period actuals
    plt.plot(test_set[time_column_name], test_set[target_column_name], c=actual_color)     # test-period actuals
    plt.plot(test_set[time_column_name], test_set['prediction'], c=pred_color)  # test-period predictions
    plt.fill_between(test_set[time_column_name].values,                      # test-period confidence interval
                 test_set['prediction_lower95'].values, # workaround for matplotlib cannot handle datetime64s
                 test_set['prediction_upper95'].values, 
                 color=pred_color, alpha=.1)
    
    plt.fill_between(test_set[time_column_name].values,                      # test-period confidence interval
                 test_set['prediction_lower50'].values, # workaround for matplotlib cannot handle datetime64s
                 test_set['prediction_upper50'].values, 
                 color=pred_color, alpha=.2)
    
    # plt.xticks(rotation=60)
    plt.legend(['Actual (train)', 'Actual (test)', 'Prediction', '95% Confidence', '50% Confidence'], 
               loc="upper left",
               fontsize='x-large')
    plt.title(' and '.join(filter_dict.values()))
    
    plt.show()
    return train_set, test_set
    
def jupyter_matplotlib_magic():
    # Uncomment me
    # %matplotlib inline 
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = [16, 8]
    import matplotlib.pyplot as plt