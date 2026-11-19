"""Whitelisted label-free value objects and completed-window compilation."""
from dataclasses import dataclass,asdict
from itertools import islice
import hashlib,hmac,json,math,warnings
import numpy as np
from .features import PredictorInput,prefix_features

@dataclass(frozen=True,slots=True)
class MonitorRecord:
    unit_id:str
    numeric_means:tuple[float|None,...]
    coverage_means:tuple[float,...]
    mean_prediction:float
    observed_rows:int
    def __post_init__(self):
        if not isinstance(self.numeric_means,tuple) or not isinstance(self.coverage_means,tuple):raise TypeError('Immutable tuples required')
        if not isinstance(self.unit_id,str) or not self.unit_id or type(self.observed_rows) is not int or len(self.numeric_means)!=40 or len(self.coverage_means)!=19 or not 1<=self.observed_rows<=24:raise ValueError('Invalid monitoring window')
        if any(v is not None and not math.isfinite(v) for v in self.numeric_means):raise ValueError('Invalid numeric summary')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in self.coverage_means):raise ValueError('Invalid coverage summary')
        if not math.isfinite(self.mean_prediction) or not 0<=self.mean_prediction<=1:raise ValueError('Invalid prediction')

@dataclass(frozen=True,slots=True)
class MonitorBatch:
    step:int
    windows:tuple[MonitorRecord,...]
    def __post_init__(self):
        if not isinstance(self.windows,tuple) or type(self.step) is not int or self.step<1 or not self.windows or any(type(w) is not MonitorRecord for w in self.windows):raise ValueError('Only label-free windows accepted')
        if len({w.unit_id for w in self.windows})!=len(self.windows):raise ValueError('Duplicate record in batch')
    def to_json(self):return json.dumps(asdict(self),allow_nan=False,sort_keys=True)
    @classmethod
    def from_json(cls,text):
        value=json.loads(text)
        if set(value)!={'step','windows'}:raise ValueError('Forbidden monitor field')
        names={'unit_id','numeric_means','coverage_means','mean_prediction','observed_rows'}
        records=[]
        for w in value['windows']:
            if set(w)!=names:raise ValueError('Forbidden record field')
            records.append(MonitorRecord(w['unit_id'],tuple(w['numeric_means']),tuple(w['coverage_means']),w['mean_prediction'],w['observed_rows']))
        return cls(value['step'],tuple(records))

@dataclass(frozen=True,slots=True)
class PredictionTrace:
    unit_id:str
    probabilities:tuple[float,...]
    def __post_init__(self):
        if not isinstance(self.probabilities,tuple) or not isinstance(self.unit_id,str) or not self.unit_id or not 1<=len(self.probabilities)<=24 or any(not math.isfinite(p) or not 0<=p<=1 for p in self.probabilities):raise ValueError('Invalid prefix prediction trace')

@dataclass(frozen=True,slots=True)
class CompiledWindow:
    monitor:MonitorRecord
    trace:PredictionTrace

def opaque_id(key,secret):
    if not isinstance(secret,bytes) or len(secret)<16:raise ValueError('A local opaque-ID secret is required')
    return hmac.new(secret,f'{key.hospital}|{key.filename}'.encode(),hashlib.sha256).hexdigest()

def compile_window(unit_id,observations,transform,predict):
    """Callbacks receive only immutable prefix inputs; no vault, key or future rows."""
    prefix=[];points=[];prob=[]
    for observation in islice(observations,24):
        if len(observation)!=40:raise ValueError('Covariates only')
        prefix.append(tuple(observation));raw=prefix_features(prefix);point=transform(raw)
        if type(point) is not PredictorInput or point.row!=raw.row:raise ValueError('Invalid frozen transformation result')
        p=float(predict(point))
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid frozen prediction')
        points.append(point.values);prob.append(p)
    if not points:raise ValueError('Empty record window')
    values=np.array(points)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);means=np.nanmean(values,axis=0)
    numeric=tuple(None if np.isnan(v) else float(v) for v in means[:40])
    record=MonitorRecord(unit_id,numeric,tuple(float(v) for v in means[40:]),float(np.mean(prob)),len(points))
    return CompiledWindow(record,PredictionTrace(unit_id,tuple(prob)))
