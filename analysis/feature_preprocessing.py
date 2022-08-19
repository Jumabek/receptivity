import pandas as pd
from copy import deepcopy


def drop_zero_variance(data):    
    M_feature = data.columns.str.contains('#')
    
    assert M_feature.sum()==data.shape[1], \
    f'Warning: data contains only features {M_feature.sum()} vs. data dim {data.shape[1]}'
    
    redundant_columns = data.columns[data.std()==0]    
    if len(redundant_columns)>0:
        print(f"{len(redundant_columns)} from total of {data.shape[1]} is/are being dropped")    
    data = data.drop(redundant_columns, axis=1)
    
    return data


# when extracting supoprt features some values of sensor will
# not occur in the data,
# hence will be instantiated as NaN when making dataframe
def impute_support_feature(df):
    support_features = df.columns[df.columns.str.contains('#SUP')]
    df[support_features] = df[support_features].fillna(0)
    return df

    
def impute_support_features_old(df_all): # probably should be done while extracting featuerfs
    df_imp = []
    for pid in (df_all.index.get_level_values('pid').unique()):
        df = df_all.loc[pid]
        support_features = df.columns[df.columns.str.contains('#SUP')]
        for col in support_features:
            df.loc[df[col].isnull(),col] = 0
        
        df_imp.append(df.assign(pid=pid))
    return pd.concat(df_imp).reset_index().set_index(df_all.index.names)


def impute(data, method='participant_mean'):
    pids = data.index.get_level_values('pid').unique()
    df_list = []
    for pid in pids: 
        pdata = data.loc[pid]
        df = impute_participant(pdata, method='participant_mean')
        #df.insert(0,'pid',pid)
        df_list.append(df.assign(pid=pid))
    return pd.concat(df_list).reset_index(        
    ).set_index(['pid','timestamp']).sort_index()


def impute_participant(pdata, method='participant_mean'):
    NAFILL=-1

    if method=='participant_mean':
        for c in pdata.columns[pdata.dtypes==float]:
        # other features such as timeseries should be imputed with mean for that partiicpant
            pdata[c].fillna(pdata[c].mean(), inplace=True)
        for c in pdata.columns[pdata.dtypes==float]:
            pdata[c] = pdata[c].fillna(NAFILL)# there will still be some NaN values
    
    for c in pdata.columns[pdata.dtypes!=float]:
    # other features such as timeseries should be imputed with mean for that partiicpant
        pdata[c].fillna(method='ffill', inplace=True)
    return pdata

import traceback

def normalization(df_orig):
    df = deepcopy(df_orig)
    I_num = df.dtypes==float    
    for c in df.columns[I_num]:
        df[c] = (df[c]-df[c].min()) / (df[c].max()-df[c].min())        
    return df

