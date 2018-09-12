def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed

def add_gdp2df(df,gdp)
    gdp_list = [np.array(gdp['GDP'][0])]
    for date in df['Date'].iloc[1:]: 
        if date in gdp['Date'].tolist():
            curr_gdp = gdp[gdp['Date'] == date]['GDP'].values
        else:
            curr_gdp = gdp_list[-1]
        gdp_list.append(curr_gdp)
    
    df['GDP'] = gdp_list
    return df