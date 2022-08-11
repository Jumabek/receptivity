import numpy as np
import pandas as pd
from typing import Iterable, Dict, List
from glob import glob

import pytz
TZ = pytz.FixedOffset(540) # GMT+09:00; Asia/Seoulsilent

import pygeohash as geo
from functools import reduce

import sys  
sys.path.insert(0, '../')
from utils import *

Log.LEVEL = 2
from sys import platform

reg_ex = '/home/juma/dataonssd/kemphone/aggregated_csv/{}_*'    
    

COLUMNS = get_columns()
DTYPES = get_dataTypes()

def replace_inf_with_minmax(df):
    df = df.replace({
        np.inf: df[np.isfinite(df)].max().max(), 
        -np.inf: df[np.isfinite(df)].min().min()
        })
    return df


def _load_data(data_source: str, pid: str, in_field: str=None, in_value: Iterable[str] = None):
    M = COLUMNS.str.contains(data_source)|COLUMNS.str.contains('timestamp')    
    try:        
        fn = glob(reg_ex.format(pid))[0]
        data = pd.read_csv(fn, usecols=COLUMNS[M], dtype=DTYPES)
        # https://docs.google.com/document/d/1dqI7F0-3st5771hrV2RIgh-mG_El5fvz/edit
        data.timestamp = pd.to_datetime( #All timestamps are represented in UTC+0.
            data.timestamp.values, unit='ms', utc=True
            ).tz_convert(TZ).tz_localize(None)
    except Exception as e:
        print('glob(reg_ex.format(pid))',glob(reg_ex.format(pid)))
        print('_load_data', 'error in read_csv', pid,e)
    if in_field and in_value:
        try:
            data = data.loc[data[in_field].isin(in_value)]
        except:
            Log.err(
                    '_load_data', 'Error occurs on pid = {}, data = {} data.columns = {} \nTraceback:\n{}'.format(pid,
                     data_source, data.columns,  traceback.format_exc()))
    return data.set_index('timestamp').dropna(how='all')


def _load_app_usage(_pid: str):
    _category = pd.read_csv(
        'app_category.csv' ,
        dtype={'AppUsageEventEntity-packageName':str, 'abstract':str,'category':str}
    )
    
    _data = _load_data(
        'AppUsageEventEntity', _pid, 'AppUsageEventEntity-type'
        , ['MOVE_TO_FOREGROUND', 'MOVE_TO_BACKGROUND']
    )
    
    _data = pd.merge(
        _data.reset_index(), _category, how='left', on='AppUsageEventEntity-packageName'
    )
    _data.set_index('timestamp', inplace=True)
    
    _app = _data.assign(
        event=lambda x: np.where(
            x['AppUsageEventEntity-type'] == 'MOVE_TO_FOREGROUND'
            , x['AppUsageEventEntity-packageName'], 'UNDEFINED'
        )
    )['event'] # which app is in the foreground

    _app_cat = _data.assign(
        event=lambda x: np.where(
            x['AppUsageEventEntity-type'] == 'MOVE_TO_FOREGROUND'
            , x['abstract'], 'UNDEFINED'
        )
    )['event'] # which category does the foreground app belong to

    return dict(
        appUsage_appPackage=_app,
        appUsage_appCategory=_app_cat
    )


def _load_battery(_pid: str):
    _data = _load_data('BatteryEntity', _pid)
    return dict(
        # convert from deciCelcious to Celcius  ,
        battery_temperature=_data['BatteryEntity-temperature']/10 
        ,battery_level=_data['BatteryEntity-level']
        , battery_plugState=_data['BatteryEntity-plugged'],# AC, Undefined, USM, wireless
        battery_status=_data['BatteryEntity-status']
    )

def _load_data_traffic(_pid: str):
    _data = _load_data('DataTrafficEntity', _pid)
    _data = _data[
            (_data['DataTrafficEntity-rxKiloBytes']>=0)&\
            (_data['DataTrafficEntity-txKiloBytes']>=0)
        ] # should not be negative
    _rx = _data['DataTrafficEntity-rxKiloBytes']
    _tx = _data['DataTrafficEntity-txKiloBytes']

    return dict(
        data_RX=_rx,
        data_TX=_tx,
        data_RX2TX = replace_inf_with_minmax(_rx/_tx)
    )


def _load_deviceEvent(_pid: str):
    _data = _load_data('DeviceEventEntity', _pid)['DeviceEventEntity-type']
    _data = _data.astype('str')
    screen = _data[_data.isin(['SCREEN_ON', 'SCREEN_OFF'])]
    unlock = _data[_data.isin(['UNLOCK'])]
    ringer = _data[_data.isin([
        'RINGER_MODE_NORMAL','RINGER_MODE_VIBRATE', 'RINGER_MODE_SILENT'
    ])]

    return dict(
        screen=screen,
        unlock = unlock, # unlock provides no information, support of 1 all the time means nothign
        ringer = ringer
        ,headsetEvent = _data[_data.isin(['HEADSET_MIC_UNPLUGGED'])]
        ,powerEvent =  _data[_data.isin(['POWER_CONNECTED','POWER_DISCONNECTED'])]
        ,batteryEvent = _data[_data.isin(['BATTER_LOW'])]
    )


def _load_location(_pid: str, remove_outlier=False):
        
    _data = _load_data('LocationEntity', _pid)
    
    if remove_outlier:
        q95 = 61 # 95% of the cases have accuracy better than 61
        _data = _data[_data['LocationEntity-accuracy']<q95] # outlier removal, more details in EDA 
        
    _cluster = _data.assign(
        cluster=lambda x: [
            geo.encode(lat, lon, precision=7)
            for lat, lon in zip(x['LocationEntity-latitude'], x['LocationEntity-longitude'])
        ]
    )['cluster']

    _data = _data.assign(
        lat_rad=lambda x: np.radians(x['LocationEntity-latitude']),
        lon_rad=lambda x: np.radians(x['LocationEntity-longitude']),
    )

    _dist = pd.concat([
        _data,
        _data.rename(lambda x: '_{}'.format(x), axis=1).shift(1)
    ], axis=1).loc[
            lambda x: (~x['_lat_rad'].isna()) & (~x['_lon_rad'].isna()), :
            ].assign(
        dist=lambda x: [
            2 * 6371000 * np.sqrt(
                np.sin((lat2 - lat1) * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) * 0.5) ** 2)
            for lat1, lon1, lat2, lon2 in zip(
                x['lat_rad'], x['lon_rad'], x['_lat_rad'], x['_lon_rad']
            )
        ]
    )['dist'] # distance between t and t-1 location point

    return dict(
        location_cluster=_cluster,
        location_distance=_dist
    )


def _load_activity(_pid: str):
    _data = _load_data(
        'PhysicalActivityTransitionEntity', _pid,
        'PhysicalActivityTransitionEntity-transitionType',
        ['ENTER_WALKING', 'ENTER_STILL', 'ENTER_IN_VEHICLE', 'ENTER_ON_BICYCLE', 'ENTER_RUNNING', 'UNNKOWN']
    )['PhysicalActivityTransitionEntity-transitionType'].str.replace('ENTER_', '')

    return dict(
        activity_event=_data
    )


def _load_wifi(_pid: str):
    _data = _load_data('WifiEntity', _pid)    
    #"[array(['88:36:6c:be:40:14', 2437, -66], dtype=object), array(['f4:d9:fb:67:90:f1', 2452, -68], dtype=object)]"
    return dict(
        wifi_numConnections = _data['WifiEntity-wifi_info'].str.count('array') # counts #wifi_connections
    )

def _load_connectivity(_pid: str):
    _data = _load_data('ConnectivityEntity', _pid)
    _data['ConnectivityEntity-type'] = _data['ConnectivityEntity-type'].replace(dict(
        MOBILE='MOBILE',
        MOBILE_DUN='MOBILE',
        UNDEFINED='DISCONNECTED',
        VPN='MOBILE',
        WIFI='WIFI'
    ))

    # convert 1.0/0.0 to True/False
    _data['ConnectivityEntity-isConnected'].replace({'1.0':'True', '0.0':'False'},inplace=True)       
    
    return dict(
        connection_type =_data[_data['ConnectivityEntity-isConnected']=='True']['ConnectivityEntity-type'], # cuz, _data['ConnectivityEntity-isConnected']=='False' case is noisy
        # and useless as they keep swtiching. more detyails in EDA
        connection_status  = _data['ConnectivityEntity-isConnected']
    )


# Smnartwartch cols
def _load_gsr(_pid: str):
    _data = _load_data('Gsr', _pid) 
    _data['Gsr-Resistance'] = replace_inf_with_minmax(1000/_data['Gsr-Resistance']) # converting kilo-ohms to Micro Siemens (refer to EDA for detailed justification) 

    return dict(
        bandGSR_resistance=_data['Gsr-Resistance']
    )

def _load_hrt(_pid: str):
    _data = _load_data('HeartRate', _pid)[['HeartRate-BPM', 'HeartRate-Quality']]
    _data = _data[_data['HeartRate-Quality']=='LOCKED'] # According to EDA ACQUIRING states are inaccurate
    return dict(
        bandHeartRate_BPM=_data['HeartRate-BPM']
    )


def _load_skin_temperature(_pid: str):
    _data = _load_data('SkinTemperature', _pid)[['SkinTemperature-Temperature']]
    _data = _data[_data['SkinTemperature-Temperature']>0]
    
    return dict(
        bandSkinTemperature_temperature = _data['SkinTemperature-Temperature']
    )

def _load_rr_interval(_pid:str):
    _data = _load_data('RRInterval', _pid)[['RRInterval-Interval']]
    
    return dict(
        bandHeartRate_RRInterval = _data['RRInterval-Interval']
    )


def _load_amb_light(_pid:str):
    _data = _load_data('AmbientLight', _pid)[['AmbientLight-Brightness']]
    _data['AmbientLight-Brightness'] = np.log(_data['AmbientLight-Brightness']+0.00005) # makes it approximately normal
    return dict(
        bandAmbientLight_brightness = _data['AmbientLight-Brightness']
    )

def _load_accelerometer(_pid: str):
    _data = _load_data('Accelerometer', _pid)
    
    return dict(
        accelerometer_X = _data['Accelerometer-X'],
        accelerometer_Y = _data['Accelerometer-Y'],
        accelerometer_Z = _data['Accelerometer-Z']
    )


def _load_calories(_pid:str):
    _data = _load_data('Calories', _pid, 'CaloriesToday')
    _cal_burned_ = _data['Calories-CaloriesToday'].diff()    
    _cal_burned_ = _cal_burned_[(_cal_burned_>=0) & (_cal_burned_<500)] # even if there is large gap betweent he two sample readings it is unlikely to burn 500 calories in the interval. 
    
    return dict(
        bandCalory_burned = _cal_burned_, # burned calory in between each sampling        
        bandCalory_burnedToday = _data['Calories-CaloriesToday']  # we calculate on the current value for the TDY features as they are accumulative
    )


def _load_steps(_pid: str):
    _data = _load_data('Pedometer', _pid, 'Pedometer-StepsToday') # SI is 1 sec.
    
    _steps = _data['Pedometer-StepsToday'].diff()
    _steps = _steps[(_steps>=0) & (_steps<=5)] # (1) negative step is irrational. (2) even the sprinter has at most 5 steps per second [more in EDA.ipynb]
    return dict(
        bandPedometer_step    =_steps,
        bandPedometer_stepsToday = _data['Pedometer-StepsToday']
    )

def _load_uv(_pid:str):
    _data = _load_data('UV', _pid)
    _data = _data[(_data['UV-UVIndexLevel']!='HIGH')]

    _uv_inc = _data['UV-UVExposureToday'] - _data['UV-UVExposureToday'].shift(1)
    _uv_inc = _uv_inc[(_uv_inc>=0)] # exposure is incremental hence should be positive and there is only one case with HIGH UVIndexLevel
    return dict(
        bandUV_indexLevel = _data['UV-UVIndexLevel'], # mostly None, but every participant has non-null cases
        bandUV_exposureToday =  _data['UV-UVExposureToday'],
        bandUV_exposure = _uv_inc
    )

def _load_pace_n_speed(_pid:str):
    _data = _load_data('Distance', _pid)

    median, std = 1.0090000000000001, 316.21040770341585
    # make sure (1) speed is not anomalous and (2) sampling interval is not too large
    # & (_data.index.to_series().diff().dt.total_seconds()<median+2*std)
    
    _data = _data[(_data['Distance-Speed']<=500)] #pls refer to EDA for more details
    return dict(
        bandDistance_pace = _data['Distance-Pace'], # 
        bandDistance_speed = _data['Distance-Speed'] # how many centimeters are covered in 1 sec1
    )


def _resample(_x: pd.Series, _until: pd.Timestamp = None):
    # using customized interval for each data soruce shoudl be considered
    _xx = _x
    if _until and _x.index.max() < _until:
        _xx[_until] = np.nan

    if _xx.dtype != float:
        return _xx.resample('1S').ffill().ffill().dropna() # F-Fill trailing NaN and drop the first NaN
    else:
        return _xx.resample('1S').mean().interpolate(method='linear').dropna()

    
def preprocess(_pid: str, _until: pd.Timestamp = None) -> Dict[str, pd.Series]:
    Log.info('preprocess', 'Start data preprocessing: pid={}'.format(_pid))
    
    funcs = [
        _load_app_usage,
        _load_battery,
        _load_data_traffic,
        _load_deviceEvent,
        _load_connectivity,
        _load_location,
        _load_activity,
        _load_wifi,
        _load_gsr,
        _load_hrt,
        _load_skin_temperature,
        _load_rr_interval,
        _load_amb_light,
        _load_accelerometer,
        _load_uv,
        _load_steps,
        _load_calories,
        _load_pace_n_speed
    ]
    
    data = [f(_pid) for f in funcs]
    data = reduce(lambda a, b: dict(a, **b), data)
    Log.info('preprocess', 'Complete to load data.')
    
    for k, v in data.items():        
        data[k]= _resample(v, _until)
    
    return data

