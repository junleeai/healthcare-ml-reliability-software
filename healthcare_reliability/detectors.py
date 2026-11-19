"""Frozen, label-free source-calibrated development detectors.

Only Phase 3 MonitorBatch/MonitorRecord values enter scoring. Bootstrap weights
are calibration calculations, never artificial monitoring batches.
"""
from dataclasses import dataclass, replace, fields, is_dataclass
import hashlib
import math
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from .monitoring import MonitorRecord,MonitorBatch
from .partitions import AccessDenied, canonical_bytes
from .detector_access import cohort_roles

METHODS=('mean_risk','feature_max','rff_mmd','missingness','prediction_ks','never_alert','rff_numeric','rff_coverage')
CONTINUOUS=tuple(range(38))+(39,)
_COHORT_PROOF=object()


def readonly(x):
    x=np.asarray(x).copy();x.setflags(write=False);return x


@dataclass(frozen=True)
class Cohort:
    stage:str
    keys:tuple
    windows:tuple
    _proof:object=None

    @classmethod
    def authorized(cls,registry,keys,windows,stage):
        roles=cohort_roles(registry);keys=tuple(keys);windows=tuple(windows)
        if (stage not in roles or stage=='verification' or not keys or len(keys)!=len(windows)
                or len(set(keys))!=len(keys) or not set(keys)<=set(roles[stage])):
            raise AccessDenied('Wrong or contaminated detector development role')
        matrix(windows)  # validate exact schema, numeric and missingness semantics
        return cls(stage,keys,windows,(_COHORT_PROOF,stage,keys,tuple(w.unit_id for w in windows)))

    @property
    def valid(self):
        return self._proof==(_COHORT_PROOF,self.stage,self.keys,tuple(w.unit_id for w in self.windows))


def matrix(windows):
    windows=tuple(windows)
    if not windows or any(type(w) is not MonitorRecord for w in windows):raise TypeError('Only whitelisted immutable MonitorRecord inputs')
    if len({w.unit_id for w in windows})!=len(windows):raise ValueError('Repeated record in monitoring input')
    x=np.array([[np.nan if v is None else v for v in w.numeric_means] for w in windows],dtype=float)
    c=np.asarray([w.coverage_means for w in windows]);p=np.asarray([w.mean_prediction for w in windows])
    if not np.isfinite(x[:,38:]).all():raise ValueError('Gender and ICULOS must be observed')
    for j in range(38):
        if np.any(~np.isfinite(x[:,j]) & (c[:,j%19]>0)):raise ValueError('Missing numeric summary contradicts observed coverage')
    return x,c,p


def weighted_ks(ref,current,wr,wc,min_reference=1):
    """Exact empirical CDF statistic, including ties; no asymptotic p-values."""
    rv=np.flatnonzero(np.isfinite(ref));cv=np.flatnonzero(np.isfinite(current))
    if len(rv)<min_reference or len(cv)==0:return np.zeros(len(wc))
    ri=rv[np.argsort(ref[rv],kind='stable')];ci=cv[np.argsort(current[cv],kind='stable')]
    grid=np.unique(np.r_[ref[ri],current[ci]])
    rn=wr[:,ri].sum(axis=1);cn=wc[:,ci].sum(axis=1)
    a=np.column_stack([np.zeros(len(wr)),np.cumsum(wr[:,ri],axis=1)])[:,np.searchsorted(ref[ri],grid,side='right')]
    b=np.column_stack([np.zeros(len(wc)),np.cumsum(wc[:,ci],axis=1)])[:,np.searchsorted(current[ci],grid,side='right')]
    a=np.divide(a,rn[:,None],out=np.zeros_like(a),where=rn[:,None]>0)
    b=np.divide(b,cn[:,None],out=np.zeros_like(b),where=cn[:,None]>0)
    scale=np.sqrt(np.divide(rn*cn,rn+cn,out=np.zeros_like(rn),where=(rn+cn)>0))
    return np.max(np.abs(a-b),axis=1)*scale


@dataclass(frozen=True)
class RFF:
    bandwidth:float
    omega:np.ndarray
    phase:np.ndarray
    reference_mean:np.ndarray

    @classmethod
    def learn(cls,z,dimensions,seed,bandwidth_seed,pairs):
        rng=np.random.default_rng(bandwidth_seed)
        a=rng.integers(0,len(z),pairs);b=rng.integers(0,len(z),pairs)
        distances=np.linalg.norm(z[a]-z[b],axis=1)
        positive=distances[distances>0];bandwidth=float(np.median(positive)) if len(positive) else 1.
        rng=np.random.default_rng(seed)
        omega=rng.normal(size=(z.shape[1],dimensions))/bandwidth;phase=rng.uniform(0,2*np.pi,dimensions)
        with threadpool_limits(limits=1):phi=np.sqrt(2/dimensions)*np.cos(z@omega+phase)
        return cls(bandwidth,readonly(omega),readonly(phase),readonly(phi.mean(axis=0)))

    def transform(self,z):
        with threadpool_limits(limits=1):return np.sqrt(2/len(self.phase))*np.cos(z@self.omega+self.phase)


@dataclass(frozen=True)
class Prepared:
    x:np.ndarray
    c:np.ndarray
    p:np.ndarray
    phi:tuple
    categories:np.ndarray


@dataclass(frozen=True)
class FrozenSuite:
    reference_ids:tuple
    reference:Prepared
    medians:np.ndarray
    scales:np.ndarray
    coverage_center:np.ndarray
    risk_sd:float
    coverage_sd:np.ndarray
    gender_values:tuple
    maps:tuple
    thresholds:tuple
    configuration:dict

    def fingerprint(self):
        # Value-based, independent of pickle order, strides and dtype memoization.
        def state(v):
            if isinstance(v,np.ndarray):
                return {'dtype':v.dtype.str,'shape':list(v.shape),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
            if is_dataclass(v):return {f.name:state(getattr(v,f.name)) for f in fields(v)}
            if isinstance(v,dict):return {k:state(x) for k,x in sorted(v.items())}
            if isinstance(v,(tuple,list)):return [state(x) for x in v]
            if isinstance(v,np.generic):return v.item()
            return v
        return hashlib.sha256(canonical_bytes(state(self))).hexdigest()

    def representations(self,x,c):
        # None is explicit unknown, filled only with a reference anchor and paired
        # with coverage. Model-imputed zeros are already standardized inputs.
        filled=np.where(np.isfinite(x),x,self.medians)
        numeric=np.clip((filled-self.medians)/self.scales,-5,5)/np.sqrt(40)
        coverage=(c-self.coverage_center)/np.sqrt(19)
        return np.column_stack([numeric,coverage])/np.sqrt(2),numeric,coverage

    def prepare(self,windows):
        x,c,p=matrix(windows)
        reps=self.representations(x,c)
        categories=np.column_stack([x[:,38]==v for v in self.gender_values]+[~np.isin(x[:,38],self.gender_values)]).astype(float)
        return Prepared(x,c,p,tuple(m.transform(z) for m,z in zip(self.maps,reps)),categories)

    def weighted_statistics(self,current,wr,wc):
        if (wr.ndim!=2 or wc.ndim!=2 or len(wr)!=len(wc) or wr.shape[1]!=len(self.reference.p)
                or wc.shape[1]!=len(current.p) or not np.isfinite(wr).all() or not np.isfinite(wc).all()
                or (wr<0).any() or (wc<0).any() or (wr.sum(axis=1)<=0).any() or (wc.sum(axis=1)<=0).any()):
            raise ValueError('Invalid whole-record calibration weights')
        nr=wr.sum(axis=1);nc=wc.sum(axis=1);r=wr/nr[:,None];c=wc/nc[:,None]
        neff=np.sqrt(nr*nc/(nr+nc))
        out={}
        with threadpool_limits(limits=1):
            out['mean_risk']=np.abs(c@current.p-r@self.reference.p)/(self.risk_sd*np.sqrt(1/nr+1/nc))
            cols=[weighted_ks(self.reference.x[:,j],current.x[:,j],wr,wc,self.configuration['ks_min_observed_reference']) for j in CONTINUOUS]
            tv=.5*np.abs(c@current.categories-r@self.reference.categories).sum(axis=1)*neff
            out['feature_max']=np.max(np.column_stack(cols+[tv]),axis=1)
            for name,j in [('rff_mmd',0),('rff_numeric',1),('rff_coverage',2)]:
                delta=c@current.phi[j]-r@self.reference.phi[j]
                out[name]=np.sum(delta*delta,axis=1)
            out['missingness']=np.max(np.abs(c@current.c-r@self.reference.c)/(self.coverage_sd[None,:]*np.sqrt(1/nr+1/nc)[:,None]),axis=1)
        out['prediction_ks']=weighted_ks(self.reference.p,current.p,wr,wc)
        out['never_alert']=np.zeros(len(wc))
        return out

    def statistics(self,windows):
        current=self.prepare(windows)
        return {k:float(v[0]) for k,v in self.weighted_statistics(current,np.ones((1,len(self.reference.p))),np.ones((1,len(windows)))).items()}

    def __call__(self,batch):
        if type(batch) is not MonitorBatch:raise TypeError('Only label-free MonitorBatch accepted')
        if set(self.reference_ids).intersection(w.unit_id for w in batch.windows):raise AccessDenied('Reference record reused as monitoring record')
        sizes={s for s,m,t in self.thresholds}
        if len(batch.windows) not in sizes:raise ValueError('No locked threshold for this batch size')
        stats=self.statistics(batch.windows);thresholds={(s,m):t for s,m,t in self.thresholds}
        return {m:{'statistic':stats[m],'threshold':thresholds[(len(batch.windows),m)],
                   'alert':False if m=='never_alert' else bool(stats[m]>thresholds[(len(batch.windows),m)])} for m in METHODS}


def fit_reference(cohort,configuration):
    if type(cohort) is not Cohort or not cohort.valid or cohort.stage!='reference':raise AccessDenied('Authorized reference-only capability required')
    x,c,p=matrix(cohort.windows)
    med=[];scale=[]
    for j in range(40):
        v=x[np.isfinite(x[:,j]),j]
        med.append(float(np.median(v)) if len(v) else 0.)
        q=float(np.subtract(*np.quantile(v,[.75,.25]))) if len(v) else 0.
        scale.append(q if q>0 else 1.)
    risk_sd=max(float(np.std(p,ddof=1)) if len(p)>1 else 0.,configuration['mean_risk_sd_floor'])
    covsd=np.maximum(np.std(c,axis=0,ddof=1) if len(c)>1 else np.zeros(19),configuration['missingness_scale_floor'])
    values=tuple(float(v) for v in np.unique(x[:,38]))
    placeholder=FrozenSuite(tuple(sorted(w.unit_id for w in cohort.windows)),None,readonly(med),readonly(scale),readonly(c.mean(axis=0)),risk_sd,readonly(covsd),values,(),(),dict(configuration))
    reps=placeholder.representations(x,c)
    maps=tuple(RFF.learn(z,configuration['rff_dimensions'],configuration['rff_seed']+j,
                        configuration['bandwidth_seed'],configuration['bandwidth_pairs']) for j,z in enumerate(reps))
    placeholder=replace(placeholder,maps=maps)
    return replace(placeholder,reference=placeholder.prepare(cohort.windows))


def bootstrap_calibration(suite,cohort,batch_size,draws,seed):
    if type(cohort) is not Cohort or not cohort.valid or cohort.stage!='calibration':raise AccessDenied('Calibration-only capability required; held-out checks forbidden')
    if set(suite.reference_ids).intersection(w.unit_id for w in cohort.windows):raise AccessDenied('Reference/calibration record overlap')
    current=suite.prepare(cohort.windows);m=len(suite.reference.p);n=len(cohort.windows)
    rng=np.random.default_rng(seed)
    wr=rng.multinomial(m,np.full(m,1/m),size=draws).astype(float)
    wc=rng.multinomial(batch_size,np.full(n,1/n),size=draws).astype(float)
    scores=suite.weighted_statistics(current,wr,wc)
    rank=min(draws,math.ceil((draws+1)*(1-suite.configuration['false_alarm_target'])))
    thresholds={k:None if k=='never_alert' else float(np.sort(v)[rank-1]) for k,v in scores.items()}
    return thresholds,scores


def lock_thresholds(suite,thresholds):
    rows=[]
    for size,by_method in sorted(thresholds.items()):
        if size not in (250,500,1000) or set(by_method)!=set(METHODS):raise ValueError('Unreviewed size or incomplete method thresholds')
        for method,t in by_method.items():
            if method!='never_alert' and (not isinstance(t,(float,int)) or not math.isfinite(t) or t<0):raise ValueError('Invalid alert threshold')
            if method=='never_alert' and t is not None:raise ValueError('Never-alert has no finite cutoff')
            rows.append((size,method,t))
    return replace(suite,thresholds=tuple(rows))


def assignment(windows,size,seed):
    matrix(windows)
    if size not in (250,500,1000):raise ValueError('Unapproved batch size')
    ordered=sorted(windows,key=lambda w:hashlib.sha256(f'phase5-assignment-v1|{seed}|{w.unit_id}'.encode()).hexdigest())
    return tuple(MonitorBatch(j//size+1,tuple(ordered[j:j+size])) for j in range(0,len(ordered)-size+1,size))
