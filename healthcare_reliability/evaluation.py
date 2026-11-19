"""Released-outcome scoring and record-cluster statistics, never detectors."""
from dataclasses import dataclass
from types import MappingProxyType
import hashlib,math
import numpy as np
from .partitions import RecordKey,Purpose,Direction,canonical_bytes,AccessDenied
from .monitoring import PredictionTrace
from .release import ReleasedOutcome

@dataclass(frozen=True,slots=True)
class RecordLoss:
    key:RecordKey
    loss_sum:float
    eligible_hours:int
    positive_hours:int
    endpoint_digest:str
    def __post_init__(self):
        if not 0<=self.positive_hours<=self.eligible_hours<=24 or not math.isfinite(self.loss_sum) or self.loss_sum<0:raise ValueError('Invalid record loss')
        if not self.eligible_hours and self.loss_sum!=0:raise ValueError('Unscorable records do not have zero assigned mean loss')
    @property
    def mean(self):return self.loss_sum/self.eligible_hours if self.eligible_hours else None

class OfflineEvaluator:
    __slots__=('_registry','_keys')
    def __init__(self,registry,unit_keys,*,allow_synthetic=False):
        if not registry.verified and not allow_synthetic:raise AccessDenied('Verified membership registry required')
        self._registry=registry;self._keys=MappingProxyType(dict(unit_keys))
    def score(self,trace,released):
        if type(trace) is not PredictionTrace or type(released) is not ReleasedOutcome:raise PermissionError('Scoring requires trace and legitimately released outcomes')
        packet=released.packet
        if trace.unit_id!=packet.unit_id or len(trace.probabilities)!=len(packet.endpoints):raise ValueError('Trace/outcome mismatch')
        key=self._keys[trace.unit_id];self._registry.authorize(key,Purpose.AUDIT,Direction.PRIMARY)
        losses=[];positive=0
        for p,e in zip(trace.probabilities,packet.endpoints):
            if not e.eligible:continue
            p=min(1-1e-6,max(1e-6,p));losses.append(-e.outcome*math.log(p)-(1-e.outcome)*math.log1p(-p));positive+=e.outcome
        digest=hashlib.sha256(canonical_bytes([(e.row,e.state.value,e.outcome) for e in packet.endpoints])).hexdigest()
        return RecordLoss(key,math.fsum(losses),len(losses),positive,digest)

@dataclass(frozen=True,slots=True)
class LossEstimate:
    records:int
    scored_records:int
    positive_records:int
    negative_only_records:int
    eligible_hours:int
    patient_balanced:float|None
    hour_weighted:float|None

def estimate(records):
    records=tuple(records)
    if len({r.key for r in records})!=len(records):raise ValueError('Repeated original record; not independent examples')
    eligible=[r for r in records if r.eligible_hours]
    return LossEstimate(len(records),len(eligible),sum(r.positive_hours>0 for r in eligible),sum(r.positive_hours==0 for r in eligible),sum(r.eligible_hours for r in eligible),math.fsum(r.mean for r in eligible)/len(eligible) if eligible else None,math.fsum(r.loss_sum for r in eligible)/sum(r.eligible_hours for r in eligible) if eligible else None)

@dataclass(frozen=True,slots=True)
class Interval:
    lower:float
    upper:float

@dataclass(frozen=True,slots=True)
class ContinuousComparison:
    patient_delta:float|None
    patient_interval:Interval|None
    hourly_delta:float|None
    hourly_interval:Interval|None
    reference:LossEstimate
    current:LossEstimate
    mode:str
    bootstrap_draws:int
    undefined_draws:int
    bootstrap_seed:int=20261117
    primary_estimand:str='patient_record_balanced_log_loss_change'


def compare(reference,current,*,mode,draws=2000,seed=20261117):
    reference=tuple(reference);current=tuple(current)
    er,ec=estimate(reference),estimate(current)
    if not reference or not current or draws<20:raise ValueError('Nonempty cohorts and at least 20 draws required')
    r={v.key:v for v in reference};c={v.key:v for v in current};shared=r.keys() & c.keys()
    if mode=='independent':
        if shared:raise ValueError('Unacknowledged record overlap')
    elif mode=='paired':
        if r.keys()!=c.keys():raise ValueError('Paired comparisons require identical record keys')
    elif mode=='overlap':
        if not shared or r.keys()==c.keys():raise ValueError('Use independent or paired mode for these cohorts')
    else:raise ValueError('Explicit comparison mode required')
    for key in shared:
        if r[key].endpoint_digest!=c[key].endpoint_digest or r[key].eligible_hours!=c[key].eligible_hours:raise ValueError('Paired feature views must retain the original endpoint/eligibility')
    # Stratify hospital and arm-membership. Shared keys get the same multiplicity
    # in both arms; fixed hospital mixture and cohort sizes are preserved.
    keys=tuple(sorted(r.keys()|c.keys()));groups={}
    for j,key in enumerate(keys):groups.setdefault((key.hospital,key in r,key in c),[]).append(j)
    rng=np.random.default_rng(seed);values=[]
    def weighted(pool,weights):
        active=[(pool[k],weights[j]) for j,k in enumerate(keys) if k in pool and weights[j]>0 and pool[k].eligible_hours]
        n=sum(w for _,w in active);hours=sum(v.eligible_hours*w for v,w in active)
        if not n or not hours:return np.array([np.nan,np.nan])
        return np.array([sum(v.mean*w for v,w in active)/n,sum(v.loss_sum*w for v,w in active)/hours])
    for _ in range(draws):
        w=np.zeros(len(keys),dtype=int)
        for indices in groups.values():
            chosen=rng.choice(indices,len(indices),replace=True);w+=np.bincount(chosen,minlength=len(keys))
        values.append(weighted(c,w)-weighted(r,w))
    values=np.array(values);valid=np.isfinite(values).all(axis=1);undefined=int((~valid).sum())
    def interval(col):
        if undefined/draws>.05 or not valid.any():return None
        lo,hi=np.quantile(values[valid,col],[.025,.975]);return Interval(float(lo),float(hi))
    dp=ec.patient_balanced-er.patient_balanced if None not in (ec.patient_balanced,er.patient_balanced) else None
    dh=ec.hour_weighted-er.hour_weighted if None not in (ec.hour_weighted,er.hour_weighted) else None
    return ContinuousComparison(dp,interval(0),dh,interval(1),er,ec,mode,draws,undefined,seed)

@dataclass(frozen=True,slots=True)
class SecondaryMarginResult:
    margin:float
    conclusion:str
    support_met:bool
    secondary_only:bool=True

def secondary_margin(comparison,*,margin):
    if margin not in {.005,.01,.02,.04}:raise ValueError('Only explicitly locked secondary sensitivity margins accepted')
    supported=all(v.scored_records>=2000 and v.positive_records>=50 and v.negative_only_records>=1000 for v in (comparison.reference,comparison.current))
    interval=comparison.patient_interval
    if not supported or interval is None:conclusion='indeterminate'
    elif interval.lower>margin:conclusion='deterioration_exceeds_margin'
    elif interval.upper<margin:conclusion='deterioration_below_margin'
    else:conclusion='indeterminate'
    return SecondaryMarginResult(margin,conclusion,supported)
