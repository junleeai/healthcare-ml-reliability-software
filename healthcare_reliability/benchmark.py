"""Lock-bound final replay interfaces. Old Phase 3 final-denial rules stay intact."""
from dataclasses import dataclass,replace,field
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,math
import numpy as np
from .partitions import RecordKey,Purpose,Direction,AccessDenied,canonical_bytes
from .features import prefix_features,COLUMNS
from .monitoring import MonitorBatch,MonitorRecord,CompiledWindow,PredictionTrace
from .release import OutcomePacket
from .endpoints import OutcomeSeries,adjudicate


_LOCK_PROOF=object()

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def object_sha(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()
def derived_seed(namespace,seed):return int(hashlib.sha256(f'phase6|{namespace}|{seed}'.encode()).hexdigest()[:8],16)


@dataclass(frozen=True)
class ExperimentLock:
    content:dict
    sha256:str
    root:Path
    _proof:object=field(default=None,repr=False)
    @classmethod
    def load(cls,path,root):
        path=Path(path);raw=path.read_bytes();v=json.loads(raw);root=Path(root)
        receipt=json.loads((root/'results/phase6/preaccess-verification.json').read_text())
        if hashlib.sha256(raw).hexdigest()!=receipt['experiment_lock_sha256']:raise AccessDenied('Unapproved or changed experiment lock')
        for p,h in v['frozen_files'].items():
            if sha(root/p)!=h:raise AccessDenied(f'Frozen component/code changed: {p}')
        if receipt['final_access_before_lock']!=0 or v['version']!='phase6-v1.0.0':raise AccessDenied('Invalid preaccess receipt')
        return cls(v,hashlib.sha256(raw).hexdigest(),root.resolve(),_LOCK_PROOF)


class FinalRegistry:
    """Explicit Phase 6 capability, never a Phase 3 final-reader bypass."""
    def __init__(self,base,lock,allowed):
        if not base.verified or type(lock) is not ExperimentLock or lock._proof is not _LOCK_PROOF or base.base_digest!=lock.content['base_manifest_sha256']:raise AccessDenied('Verified lock and memberships required')
        self.base=base;self.lock=lock;self.allowed=frozenset(allowed);self.stage='covariates';self.active=False;self.teacher_unlocked=False
        if object_sha([f'{k.hospital}/{k.filename}' for k in sorted(self.allowed)])!=lock.content['assigned_record_union_sha256']:raise AccessDenied('Final roster differs from lock')
        if any(base.original_role(k)!='final_evaluation' for k in self.allowed):raise AccessDenied('Development record in final capability')
    def unlock_teacher(self,path):
        seal=json.loads(Path(path).read_text())
        if self.stage!='covariates' or seal['experiment_sha256']!=self.lock.sha256 or seal['replays']!=self.lock.content['monitoring_replays'] or seal['labels_read_before_seal']!=0:
            raise AccessDenied('Incomplete or wrong pre-outcome monitoring seal')
        if sha(Path(path).parent/'monitoring-records.joblib')!=seal['monitoring_records_sha256']:raise AccessDenied('Monitor store changed')
        self.teacher_unlocked=True;self.stage='teacher'
    @property
    def verified(self):return self.base.verified
    @property
    def base_digest(self):return self.base.base_digest
    def keys(self):return self.base.keys()
    def original_role(self,key):return self.base.original_role(key)
    def authorize(self,key,purpose,direction=Direction.PRIMARY,heldout_fold=None):
        if not self.active:raise AccessDenied('No active lock-bound final file scope')
        if key not in self.allowed or direction is not Direction.PRIMARY:raise AccessDenied('Outside locked final roster/direction')
        if purpose is Purpose.REPLAY and self.stage=='covariates':return
        if purpose is Purpose.AUDIT and self.teacher_unlocked and self.stage in ('teacher','replay','analysis'):return
        raise AccessDenied('Final capability rejects purpose/stage')


@contextmanager
def final_file_tripwire(raw_root,registry):
    root=Path(raw_root).resolve();original=Path.open
    if root!=(registry.lock.root/'data/raw').resolve():raise AccessDenied('Raw root differs from lock root')
    registry.active=True
    audit={'opened':set(),'opens':{},'blocked':0}
    def guarded(path,*args,**kwargs):
        p=path.resolve()
        if p.suffix=='.psv' and root in p.parents:
            rel=p.relative_to(root);key=RecordKey(*rel.parts) if len(rel.parts)==2 else None
            mode=args[0] if args else kwargs.get('mode','r')
            if key not in registry.allowed or any(c in mode for c in 'wax+') or registry.stage not in ('covariates','teacher','integrity'):
                audit['blocked']+=1;raise AccessDenied('Unscoped final/development/raw-write attempt')
            audit['opened'].add(key);audit['opens'][registry.stage]=audit['opens'].get(registry.stage,0)+1
        return original(path,*args,**kwargs)
    Path.open=guarded
    try:yield audit
    finally:Path.open=original;registry.active=False


def compile_view(unit,observations,model):
    obs=tuple(observations)
    if not 1<=len(obs)<=24:raise ValueError('Window exceeds frozen candidate prefix')
    x=np.asarray([prefix_features(obs[:r]).values for r in range(1,len(obs)+1)])
    z=model.preprocessor.transform(x);p=model.predict_transformed_matrix(z)
    return CompiledWindow(MonitorRecord(unit,tuple(float(v) for v in z[:,:40].mean(axis=0)),tuple(float(v) for v in z[:,40:].mean(axis=0)),float(p.mean()),len(obs)),PredictionTrace(unit,tuple(float(v) for v in p)))


def perturb(observations,key,scenario,model,lock):
    x=np.asarray(observations).copy();kind=scenario['kind']
    if kind=='missingness':
        cols=[COLUMNS.index(n) for n in ('Glucose','Creatinine','Platelets')]
        rng=np.random.default_rng(derived_seed(f'missingness|{key.hospital}|{key.filename}',lock['perturbation_seed']))
        u=rng.random((len(x),3));part=x[:,cols].copy();part[u<scenario['intensity']]=np.nan;x[:,cols]=part
    elif kind=='hr_offset':
        j=COLUMNS.index('HR');x[:,j]+=scenario['intensity']*model.preprocessor.scales[0]
    elif kind=='unused_ast':x[:,COLUMNS.index('AST')]+=3*model.preprocessor.scales[17]
    elif kind=='unused_age':x[:,COLUMNS.index('Age')]+=10
    elif kind not in ('none','transfer','conditional'):raise ValueError('Unreviewed intervention')
    if not np.array_equal(x[:,39],np.asarray(observations)[:,39]):raise ValueError('Intervention changed time')
    return x


def teacher_packet(root,registry,key,unit):
    registry.authorize(key,Purpose.AUDIT,Direction.PRIMARY)
    if registry.stage!='teacher':raise AccessDenied('Endpoint adjudication is teacher-only, after monitor seal')
    times=[];labels=[]
    with (Path(root)/key.hospital/key.filename).open() as f:
        if tuple(f.readline().rstrip().split('|'))!=(*COLUMNS,'SepsisLabel'):raise ValueError('Unexpected raw schema')
        for line in f:
            fields=line.rstrip().split('|')
            if len(fields)!=41:raise ValueError('Bad row width')
            times.append(float(fields[39]));value=float(fields[40])
            if value not in (0,1):raise ValueError('Invalid annotation')
            labels.append(int(value))
    return OutcomePacket.from_adjudication(unit,adjudicate(OutcomeSeries(tuple(times),tuple(labels))))


def counterfactual_packets(keys,packets,traces,fraction,seed):
    """F-v1: whole trajectories, mask-matched strata, teacher pathway only."""
    out=dict(packets);groups={}
    selected=sorted(keys,key=lambda k:hashlib.sha256(f'F-select|{seed}|{k.hospital}|{k.filename}'.encode()).hexdigest())[:math.floor(len(keys)*fraction)]
    for k in selected:
        packet=packets[k];mask=tuple(e.eligible for e in packet.endpoints)
        reasons=tuple(e.state.value if not e.eligible else 'eligible' for e in packet.endpoints)
        groups.setdefault((len(packet.endpoints),mask,reasons),[]).append(k)
    changed=0;donors={}
    for (_,mask,_),members in groups.items():
        def risk(k):
            values=[p for p,e in zip(traces[k].probabilities,packets[k].endpoints) if e.eligible]
            return float(np.mean(values)) if values else 0.
        recipients=sorted(members,key=lambda k:(risk(k),object_sha(('recipient',k.hospital,k.filename,seed))))
        donors_sorted=sorted(members,key=lambda k:(-sum(e.outcome==1 for e in packets[k].endpoints),object_sha(('donor',k.hospital,k.filename,seed))))
        for recipient,donor in zip(recipients,donors_sorted):
            old=packets[recipient]
            endpoints=tuple(b if a.eligible else a for a,b in zip(old.endpoints,packets[donor].endpoints))
            if tuple(e.eligible for e in endpoints)!=tuple(e.eligible for e in old.endpoints):raise AssertionError('F changed eligibility')
            if endpoints!=packets[donor].endpoints:raise AssertionError('F must copy an entire existing endpoint trajectory')
            out[recipient]=OutcomePacket(old.unit_id,endpoints);donors[recipient]=donor
            changed+=endpoints!=old.endpoints
    before=sum(e.outcome==1 for k in keys for e in packets[k].endpoints)
    after=sum(e.outcome==1 for k in keys for e in out[k].endpoints)
    if before!=after:raise AssertionError('F failed event-hour preservation')
    return out,dict(selected=len(selected),changed_records=int(changed),strata=len(groups),positive_hours_preserved=True,eligibility_preserved=True),donors


@dataclass(frozen=True)
class PresealedMonitor:
    inputs:tuple
    signals:tuple
    def __call__(self,batch):
        if type(batch) is not MonitorBatch or not 1<=batch.step<=len(self.inputs):raise TypeError('Whitelisted locked batch required')
        digest=hashlib.sha256(batch.to_json().encode()).hexdigest()
        if digest!=self.inputs[batch.step-1]:raise AccessDenied('Monitoring input differs from pre-outcome seal')
        return json.loads(self.signals[batch.step-1])
