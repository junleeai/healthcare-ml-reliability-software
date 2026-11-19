"""Covariate-only current/past prefix features; no learned preprocessing."""
from dataclasses import dataclass
from itertools import islice
import math,warnings
import numpy as np

COLUMNS=('HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS')
CLINICAL=('HR','O2Sat','Temp','SBP','MAP','DBP','Resp','Hct','Hgb','WBC','Platelets','BUN','Creatinine','Glucose','Potassium','Calcium','PTT','AST','Bilirubin_total')
INDEX=tuple(COLUMNS.index(v) for v in CLINICAL)

@dataclass(frozen=True,slots=True)
class PredictorInput:
    row:int
    values:tuple[float,...]
    def __post_init__(self):
        if not isinstance(self.values,tuple) or type(self.row) is not int or not 1<=self.row<=24 or len(self.values)!=59:raise ValueError('Invalid prediction view')
        if any(math.isinf(v) for v in self.values):raise ValueError('Infinite input')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in self.values[40:]):raise ValueError('Coverage must remain an observed fraction')

def prefix_features(prefix)->PredictorInput:
    x=np.asarray(prefix,dtype=float)
    if x.ndim!=2 or x.shape[1]!=40 or not 1<=len(x)<=24:raise ValueError('Only a nonempty current/past covariate prefix, at most 24 rows, is accepted')
    if np.isinf(x).any():raise ValueError('Infinite covariate')
    window=x[-6:][:,INDEX];observed=np.isfinite(window)
    positions=np.where(observed,np.arange(len(window))[:,None],-1).max(axis=0)
    last=window[np.maximum(positions,0),np.arange(19)].copy();last[positions<0]=np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(window,axis=0)
    return PredictorInput(len(x),tuple(np.r_[last,median,x[-1,35],x[-1,39],observed.mean(axis=0)]))

@dataclass(frozen=True,slots=True)
class KeyedFeature:
    key:object
    point:PredictorInput
    def __post_init__(self):
        from .partitions import RecordKey
        if type(self.key) is not RecordKey or type(self.point) is not PredictorInput:raise TypeError('Registered key and immutable predictor input required')

def fit_preprocessor(transformer,frames,registry,heldout_fold,direction,*,allow_synthetic=False):
    """Guard a future learned transformation; never called on real data in Phase 3."""
    from .partitions import Purpose,AccessDenied
    if not registry.verified and not allow_synthetic:raise AccessDenied('Verified membership registry required')
    frames=tuple(frames)
    if not frames:raise ValueError('No fitting data')
    if any(type(frame) is not KeyedFeature for frame in frames):raise TypeError('Only keyed feature frames accepted')
    if len({(frame.key,frame.point.row) for frame in frames})!=len(frames):raise ValueError('Duplicate patient-record hour')
    for frame in frames:registry.authorize(frame.key,Purpose.TRAIN,direction,heldout_fold)
    # Metadata/record identity and targets are not numeric input coordinates.
    return transformer.fit(np.array([frame.point.values for frame in frames],dtype=float))
