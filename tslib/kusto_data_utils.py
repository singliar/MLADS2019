# Utilities for pulling and parsing data from Kusto

"""
We can access the Application Insights using Kusto query. The REST request to the application insights needs to have form as following:
```
POST https://api.applicationinsights.io/{api_version}/apps/{app_id}/query

'X-Api-Key': {api_key}
Content-Type: application/json

{
   "query": "requests * | where TimeGenerated > ago(1d) | summarize count() by appName",
}
```

To get an ```app_id``` browse to AppInsights resource and click API Access. In the API access section click "Create API Key". In the "Create API key" blade check only "Read telemetry" check box and give the name to ```api_key```.<BR/>**Note:** After the key was generated and shown, its value needs to be copied, because it will be impossible to get it after the blade is closed!<BR/>
The ```api_version``` used in this example is "v1".
For more information see the [documentation](https://dev.applicationinsights.io/documentation/Authorization/API-key-and-App-ID) page.
"""

class KustoTimeSeries:
    """Kusto time series constants."""
    TABLES = 'tables'
    ROWS = 'rows'
    COLUMNS = 'columns'
    NAME = 'name'

def send_app_insights_kusto_query(app_id:str, api_key:str, query:str, api_version='v1'):
    """
    Execute query on the Application Insights instance.
    ValueError is raised if the error code was returned from server.
    
    :param app_id: The application ID.
    :type app_id: str
    :param api_key: API key.
    :type api_key: str
    :param query: The Kusto query.
    :type query: str
    :raises: ValueError
    :returns: The response from AppInsights service.
    :rtype: str
    
    """
    rq = requests.Session()
    rq.headers.update({'X-Api-Key': api_key})
    rq.headers.update({'Content-Type': 'application/json'})
    json_string = json.dumps({'query': query})
    url_string = 'https://api.applicationinsights.io/{}/apps/{}/query'.format(api_version, app_id)
    print(url_string)
    response = rq.post(url_string, json_string)
    if response.status_code!=200:
        err_dict = json.loads(response.text)
        message = err_dict.get('error', {'message': '<empty>'}).get('message')
        raise ValueError("The request returned error {}.\nmessage: {}".format(response.status_code,
                                                                                message))
    return response.text


def kusto_json_to_data_frame(data:str, time_colname:str, grains:list=None):
    """
    Convert the time series data returned from Kusto to the pandas data frame.
    If time_colname is not present in returned data, the ValueError is raised.

    :param data: The JSON data string returned from the Kusto.
    :type data: str
    :param time_colname: The timestamp column.
    :type time_colname: str
    :param grains: If the data frame contains more than one series, it needs to have the grain columns,
                   to separate them.
    :type grains: list
    :returns: The pandas data frame with the time series.
    :rtype: pd.DataFrame
    :raises: ValueError
    
    """
    if grains is None:
        grains = []
    pd_list = []
    data = json.loads(data)
    single_table=data[KustoTimeSeries.TABLES][0]
    for i in range(len(single_table[KustoTimeSeries.ROWS])):
        dict_one_grain = {}
        for j in range(len(single_table[KustoTimeSeries.COLUMNS])):
            if single_table[KustoTimeSeries.COLUMNS][j][KustoTimeSeries.NAME] not in grains:
                try:
                    one_series = single_table[KustoTimeSeries.ROWS][i][j]
                    lst_values = json.loads(one_series)
                except:
                    print('Could not parse ' + one_series)
                dict_one_grain[single_table[KustoTimeSeries.COLUMNS][j][KustoTimeSeries.NAME]] = lst_values
        if not time_colname in dict_one_grain.keys():
            raise ValueError("The {} is absent from the returned data.".format(time_colname))
        ts_len = len(dict_one_grain[time_colname])
        for grain in grains:
            dict_one_grain[grain] = np.repeat(single_table[KustoTimeSeries.ROWS][i][0], ts_len)
        pd_list.append(pd.DataFrame(dict_one_grain))
        whole_data = pd.concat(pd_list)
        whole_data[time_colname] = pd.to_datetime(whole_data[time_colname])
    return whole_data