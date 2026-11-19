"""Phase 6 released-label-only statistics; continuous patient loss is primary."""
from dataclasses import dataclass,asdict
import numpy as np
from .evaluation import estimate,RecordLoss
from .model_development import rank_statistics,calibration,threshold_stats


def loss_change(reference,current,*,mode,seed,draws=2000,mask_identity=False):
    r={v.key:v for v in reference};c={v.key:v for v in current}
    if len(r)!=len(reference) or len(c)!=len(current):raise ValueError('Duplicate record loss')
    shared=r.keys()&c.keys()
    if mode=='independent' and shared:raise ValueError('Independent arms overlap')
    if mode in ('paired','paired_counterfactual') and r.keys()!=c.keys():raise ValueError('Paired keys differ')
    if mode=='overlap' and (not shared or r.keys()==c.keys()):raise ValueError('Wrong overlap mode')
    if mode not in ('independent','paired','overlap','paired_counterfactual'):raise ValueError('Unreviewed comparison mode')
    for k in shared:
        if r[k].eligible_hours!=c[k].eligible_hours:raise ValueError('Eligibility counts changed')
        if r[k].endpoint_digest!=c[k].endpoint_digest and not (mode=='paired_counterfactual' and mask_identity):
            raise ValueError('Clinical paired endpoint digest changed')
    if mode=='paired_counterfactual' and not mask_identity:raise ValueError('Counterfactual requires verified mask identity')
    keys=sorted(r.keys()|c.keys());groups={}
    ar=np.zeros((len(keys),3));ac=np.zeros_like(ar)
    for j,k in enumerate(keys):
        groups.setdefault((k.hospital,k in r,k in c),[]).append(j)
        for pool,a in ((r,ar),(c,ac)):
            v=pool.get(k)
            if v and v.eligible_hours:a[j]=[v.mean,v.loss_sum,v.eligible_hours]
    er,ec=estimate(reference),estimate(current)
    base=dict(reference=asdict(er),current=asdict(ec),mode=mode,seed=seed,draws=draws,
              primary_estimand='patient_record_balanced_log_loss_change',counterfactual_only=mode=='paired_counterfactual')
    if er.patient_balanced is None or ec.patient_balanced is None:
        return dict(base,patient_delta=None,hour_delta=None,patient_interval=None,hour_interval=None,undefined_draws=draws)
    dp=ec.patient_balanced-er.patient_balanced;dh=ec.hour_weighted-er.hour_weighted
    if r.keys()==c.keys() and all(r[k]==c[k] for k in r):
        return dict(base,patient_delta=dp,hour_delta=dh,patient_interval=[0.,0.],hour_interval=[0.,0.],undefined_draws=0)
    rng=np.random.default_rng(seed);values=np.empty((draws,2))
    for b in range(draws):
        weights=np.zeros(len(keys),dtype=int)
        for indices in groups.values():weights+=np.bincount(rng.choice(indices,len(indices),replace=True),minlength=len(keys))
        def weighted(a):
            n=np.dot(weights,a[:,2]>0);hours=np.dot(weights,a[:,2])
            if n==0 or hours==0:return np.array([np.nan,np.nan])
            return np.array([np.dot(weights,a[:,0])/n,np.dot(weights,a[:,1])/hours])
        values[b]=weighted(ac)-weighted(ar)
    valid=np.isfinite(values).all(axis=1);undefined=int((~valid).sum())
    ci=np.quantile(values[valid],[.025,.975],axis=0) if undefined/draws<=.05 and valid.any() else None
    return dict(base,patient_delta=dp,hour_delta=dh,patient_interval=None if ci is None else ci[:,0].tolist(),
                hour_interval=None if ci is None else ci[:,1].tolist(),undefined_draws=undefined)


def margins(comparison):
    supported=all(v['scored_records']>=2000 and v['positive_records']>=50 and v['negative_only_records']>=1000 for v in (comparison['reference'],comparison['current']))
    interval=comparison['hour_interval']
    return {str(m):dict(estimand='hour_weighted_log_loss_change',support_met=supported,
              conclusion='indeterminate' if not supported or interval is None else
              'deterioration_exceeds_margin' if interval[0]>m else
              'deterioration_below_margin' if interval[1]<m else 'indeterminate') for m in (.005,.01,.02,.04)}


@dataclass
class ReleasedHours:
    keys:tuple
    group:np.ndarray
    rows:np.ndarray
    y:np.ndarray
    counts:np.ndarray
    probabilities:np.ndarray
    @property
    def weights(self):return 1/self.counts[self.group]


def performance(keys,traces,packets,threshold):
    group=[];rows=[];y=[];p=[];counts=[]
    for j,(key,trace,packet) in enumerate(zip(keys,traces,packets,strict=True)):
        n=0
        if trace.unit_id!=packet.unit_id or len(trace.probabilities)!=len(packet.endpoints):raise ValueError('Released trace/packet mismatch')
        for prob,e in zip(trace.probabilities,packet.endpoints):
            if e.eligible:group.append(j);rows.append(e.row);y.append(e.outcome);p.append(prob);n+=1
        counts.append(n)
    d=ReleasedHours(tuple(keys),np.asarray(group,dtype=int),np.asarray(rows),np.asarray(y),np.asarray(counts),np.asarray(p))
    pos=np.bincount(d.group,weights=d.y,minlength=len(keys));scored=d.counts>0
    support=dict(records=len(keys),scored_records=int(scored.sum()),positive_records=int((pos>0).sum()),negative_only_records=int(((pos==0)&scored).sum()),
                 eligible_hours=len(y),positive_hours=int(sum(y)),negative_hours=len(y)-int(sum(y)))
    if not y:return dict(support=support,patient_balanced_log_loss=None,hour_log_loss=None)
    w=d.weights;prob=np.clip(d.probabilities,1e-6,1-1e-6)
    ll=-(d.y*np.log(prob)+(1-d.y)*np.log1p(-prob))
    auroc,ap=rank_statistics(d.y,d.probabilities,w);ha,hap=rank_statistics(d.y,d.probabilities,np.ones(len(y)))
    return dict(support=support,patient_balanced_log_loss=float(np.sum(w*ll)/scored.sum()),hour_log_loss=float(ll.mean()),
                weighted_auroc=auroc,weighted_average_precision=ap,weighted_prevalence=float(np.sum(w*d.y)/scored.sum()),
                hour_auroc=ha,hour_average_precision=hap,hour_prevalence=float(d.y.mean()),
                calibration=calibration(d,d.probabilities),operating=threshold_stats(d,d.probabilities,threshold),threshold=threshold)
