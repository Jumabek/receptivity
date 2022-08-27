import ray
import numpy as np
import pandas as pd

from os.path import join
import ntpath


SECONDS_IN_MIN = 60
MIN_IN_HOUR = 60
HOURS_IN_A_DAY = 24
COLLECTION_HOURS = 12
COLLECTION_DAYS = 7

MS_IN_SEC = 1000
SEC_IN_MIN = 60
MIN_IN_HOUR = 60
HOUR_IN_DAY = 24

MS_IN_MIN = MS_IN_SEC* SEC_IN_MIN 
MS_IN_DAY = HOUR_IN_DAY*MIN_IN_HOUR*SEC_IN_MIN*MS_IN_SEC # day    = 24 hours x 60 minutes x 60 seconds x 1000 milliseconds
SEC_IN_HOUR = SEC_IN_MIN*MIN_IN_HOUR
SEC_IN_DAY = SEC_IN_HOUR*HOUR_IN_DAY

STUDY_DURATION_IN_DAYS = 7
STUDY_DURATION = (STUDY_DURATION_IN_DAYS-1)*MS_IN_DAY+ 12*60*60*1000 # from [10 am - 22 pm] 
DAY_IN_WEEK = 7
DATAROOT = '../data/kemophone'
RANDOM_STATE = 42

def extract_pcode(fn):
    bname = ntpath.basename(fn)
    return bname[:bname.find('_')]

def get_dataTypes():
    DTYPES = {

        'Accelerometer-Y': float, 'Accelerometer-X': float, 'Accelerometer-Z': float,

        'AmbientLight-Brightness': float, 

        'AppUsageEventEntity-isSystemApp':'category'
        , 'AppUsageEventEntity-isUpdatedSystemApp':'category'
        , 'AppUsageEventEntity-packageName':str,
        'AppUsageEventEntity-type':'category', 'AppUsageEventEntity-name':str,
        'AppUsageStatEntity-endTime':float,
        'AppUsageStatEntity-isSystemApp':float,
        'AppUsageStatEntity-isUpdatedSystemApp':float,
        'AppUsageStatEntity-lastTimeUsed':int,
        'AppUsageStatEntity-name':'category',
        'AppUsageStatEntity-packageName':'category',
        'AppUsageStatEntity-startTime':'int64',
        'AppUsageStatEntity-totalTimeForeground':'int64',
                    
        'BatteryEntity-level': float,  'BatteryEntity-status': 'category', 
        'BatteryEntity-temperature': float,  'BatteryEntity-plugged': 'category',

        'CallLogEntity-isStarred':'category', 'CallLogEntity-isPinned':'category'
        , 'CallLogEntity-dataUsage':float, 'CallLogEntity-duration':float, 
        'CallLogEntity-timesContacted':float,  'CallLogEntity-presentation':'category'
        , 'CallLogEntity-type':'category', 'CallLogEntity-number':'category',
        'CallLogEntity-contact':'category',

        'Calories-CaloriesToday': float, 'Calories-Calories': float, 

        'ConnectivityEntity-isConnected':'category'
        , 'ConnectivityEntity-type':'category',        

        'DataTrafficEntity-rxKiloBytes': float, 'DataTrafficEntity-duration': float
        , 'DataTrafficEntity-txKiloBytes': float,        

        'DeviceEventEntity-type': 'category', 

        'Distance-TotalDistance': float, 'Distance-DistanceToday': float
        ,  'Distance-Speed': float, 'Distance-Pace': float
        , 'Distance-MotionType': 'category'  

        ,'Gsr-Resistance': float,  

        'HeartRate-Quality': str, 'HeartRate-BPM': float
        
        ,'LocationEntity-longitude': float, 'LocationEntity-speed': float, 'LocationEntity-latitude': float, 'LocationEntity-accuracy': float, 'LocationEntity-altitude': float,

        'MediaEntity-bucketDisplay':'category', 'MediaEntity-mimetype':'category',
        'MessageEntity-isPinned':'category', 'MessageEntity-timesContacted':float,
        'MessageEntity-contact':'category', 'MessageEntity-number':'category',
        'MessageEntity-messageBox':'category', 'MessageEntity-messageClass':'category',
        'MessageEntity-isStarred':'category',

        'Pedometer-TotalSteps': float, 'Pedometer-StepsToday': float,
        'PhysicalActivityEventEntity-confidence': float
        , 'PhysicalActivityEventEntity-type': 'category', 
        'PhysicalActivityTransitionEntity-transitionType': 'category',       

        'RecordEntity-channelMask':'category',
        'RecordEntity-duration':float,
        'RecordEntity-encoding':'category',
        'RecordEntity-path':str,
        'RecordEntity-sampleRate':float,          
                    
        'RRInterval-Interval': float,       

        'SkinTemperature-Temperature': float,                    
        'UV-UVIndexLevel': 'category', 'UV-UVExposureToday': float,          
        'WifiEntity-wifi_info': str
    }
    return DTYPES


def get_columns():
    COLUMNS = ['timestamp', 'DeviceEventEntity-type', 'UV-UVExposureToday',
       'UV-UVIndexLevel', 'WifiEntity-wifi_info', 'BatteryEntity-status',
       'BatteryEntity-plugged', 'BatteryEntity-temperature',
       'BatteryEntity-level', 'Accelerometer-Y', 'Accelerometer-X',
       'Accelerometer-Z', 'Pedometer-StepsToday', 'Pedometer-TotalSteps',
       'DataTrafficEntity-duration', 'DataTrafficEntity-txKiloBytes',
       'DataTrafficEntity-rxKiloBytes', 'AmbientLight-Brightness',
       'LocationEntity-accuracy', 'LocationEntity-altitude',
       'LocationEntity-latitude', 'LocationEntity-speed',
       'LocationEntity-longitude', 'PhysicalActivityEventEntity-type',
       'PhysicalActivityEventEntity-confidence',
       'PhysicalActivityTransitionEntity-transitionType', 'Gsr-Resistance',
       'HeartRate-BPM', 'HeartRate-Quality', 'ConnectivityEntity-isConnected',
       'ConnectivityEntity-type', 'Distance-MotionType', 'Distance-Speed',
       'Distance-Pace', 'Distance-TotalDistance', 'Distance-DistanceToday',
       'SkinTemperature-Temperature', 'Calories-CaloriesToday',
       'Calories-Calories', 'RRInterval-Interval', 'AppUsageEventEntity-name',
       'AppUsageEventEntity-type', 'AppUsageEventEntity-isUpdatedSystemApp',
       'AppUsageEventEntity-isSystemApp', 'AppUsageEventEntity-packageName',
       'CallLogEntity-isPinned', 'CallLogEntity-dataUsage',
       'CallLogEntity-timesContacted', 'CallLogEntity-presentation',
       'CallLogEntity-type', 'CallLogEntity-isStarred',
       'CallLogEntity-duration', 'CallLogEntity-number',
       'CallLogEntity-contact', 'MediaEntity-bucketDisplay',
       'MediaEntity-mimetype']
    return pd.Series(COLUMNS)


def get_all_columns():
    import numpy as np
    sample_fn = join(DATAROOT,'aggregated_csv','3028_3409214.csv') # 3028 is the participant who has all columns
    df_ = pd.read_csv(sample_fn,nrows=1)

    # exc 
    not_analyzed_sources = ['AppUsageStatEntity', 'RecordEntity'] 

    M = np.zeros_like(df_.columns, dtype=bool)
    for s in not_analyzed_sources:
        M=M|df_.columns.str.startswith(s)

    COLUMNS_EXTENDED= df_.loc[:,~M].columns
    return COLUMNS_EXTENDED


import datetime as dt
import ray

class Log:
    LEVEL = 0
    
    @staticmethod
    def _log(tag: any, msg: str=''):
        t = dt.datetime.now().strftime('%y/%m/%d %H:%M:%S')
        print('[{}] [{}] {}'.format(t, tag, msg))
    
    @staticmethod
    def info(tag: any, msg: str=''):
        if -1 < Log.LEVEL <= 0:
            Log._log(tag, msg)
        
    @staticmethod
    def warn(tag: any, msg: str=''):
        if -1 < Log.LEVEL <= 1:
            Log._log(tag, msg)
    
    @staticmethod
    def err(tag: any, msg: str=''):
        if -1 < Log.LEVEL <= 2:
            Log._log(tag, msg)      

            
class on_ray:
    def __init__(self, **kwargs):
        self._param = kwargs
    
    def __enter__(self):
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(**self._param)
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        ray.shutdown()