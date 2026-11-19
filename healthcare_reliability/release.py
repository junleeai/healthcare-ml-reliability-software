"""Outcome release is a separate channel, ordered after each monitor decision."""
from dataclasses import dataclass,field
from .endpoints import Endpoint,Adjudication
from .monitoring import MonitorBatch
from .partitions import canonical_bytes
import hashlib,json

_AFTER_MONITOR=object()
_RELEASE_PROOF=object()
_SEAL_PROOF=object()

@dataclass(frozen=True,slots=True)
class OutcomePacket:
    unit_id:str
    endpoints:tuple[Endpoint,...]
    @classmethod
    def from_adjudication(cls,unit_id,value):
        if type(value) is not Adjudication:raise TypeError('Retrospective adjudication required')
        return cls(unit_id,value.endpoints)
    def __post_init__(self):
        if not isinstance(self.endpoints,tuple) or any(type(e) is not Endpoint for e in self.endpoints) or len(self.endpoints)>24:raise TypeError('Immutable endpoint sequence required')
        if not isinstance(self.unit_id,str) or not self.unit_id or not self.endpoints or tuple(e.row for e in self.endpoints)!=tuple(range(1,len(self.endpoints)+1)):raise ValueError('Invalid outcome packet')

@dataclass(frozen=True,slots=True)
class ReleasedOutcome:
    packet:OutcomePacket
    batch_step:int
    due_step:int
    release_step:int
    _proof:object=field(repr=False,compare=False)
    def __post_init__(self):
        if type(self.packet) is not OutcomePacket or self._proof is not _RELEASE_PROOF or self.batch_step<1 or self.due_step<self.batch_step or self.release_step<self.due_step:raise PermissionError('Only due outcomes may enter scoring')

class ReleaseScheduler:
    __slots__=('_delay','_queue','_seen')
    def __init__(self,delay=3):
        if type(delay) is not int or delay not in {0,3,6}:raise ValueError('Protocol delays are 0, 3 or 6 artificial steps')
        self._delay=delay;self._queue=[];self._seen=set()
    @property
    def delay(self):return self._delay
    def stage(self,batch_step,packets):
        packets=tuple(packets)
        if any(type(p) is not OutcomePacket for p in packets):raise TypeError('Only immutable endpoint packets accepted')
        ids=[p.unit_id for p in packets]
        if type(batch_step) is not int or batch_step<1 or len(set(ids))!=len(ids) or self._seen.intersection(ids):raise ValueError('Repeated outcome record')
        self._seen.update(ids)
        self._queue.extend((batch_step,batch_step+self.delay,p) for p in packets)
    def _release(self,step,permit):
        if permit is not _AFTER_MONITOR:raise PermissionError('Release requires completed monitor decision')
        due=[(b,d,p) for b,d,p in self._queue if d<=step]
        self._queue=[v for v in self._queue if v[1]>step]
        return tuple(ReleasedOutcome(p,b,d,step,_RELEASE_PROOF) for b,d,p in due)
    @property
    def pending(self):return len(self._queue)

class FrozenStateError(RuntimeError):pass

@dataclass(frozen=True,slots=True)
class StepResult:
    step:int
    signal_json:str
    released:tuple[ReleasedOutcome,...]
    assignment_seed:int=20261118
    monitor_input_sha256:str=''
    @property
    def sensitivity_simulation(self):return self.assignment_seed!=20261118


@dataclass(frozen=True,slots=True)
class AlertSnapshot:
    step:int
    assignment_seed:int
    signal_json:str
    monitor_input_sha256:str

@dataclass(frozen=True,slots=True)
class SealedReplay:
    assignment_seed:int
    alerts:tuple[AlertSnapshot,...]
    sha256:str
    _proof:object=field(repr=False,compare=False)
    protocol_sha256:str|None=None
    def __post_init__(self):
        if self._proof is not _SEAL_PROOF:raise PermissionError('Only a successfully closed replay can be sealed')

class ReplayLifecycle:
    """No fit/update/recalibrate method and no outcome argument to the monitor."""
    __slots__=('_monitor','_review','_fingerprints','_baseline','_scheduler','_step','_seen','_closed','_batch_size','_max_steps','_expected_units','_assignment_seed','_alerts','_failed','_protocol_sha256')
    def __init__(self,monitor,review,fingerprints,delay=3,*,batch_size=500,max_steps=None,expected_units=None,assignment_seed=20261118,protocol_sha256=None):
        if batch_size<1 or (max_steps is not None and max_steps<1):raise ValueError('Invalid replay dimensions')
        self._protocol_sha256=protocol_sha256
        self._batch_size=batch_size;self._max_steps=max_steps;self._expected_units=expected_units;self._assignment_seed=assignment_seed
        self._monitor=monitor;self._review=review;self._fingerprints=fingerprints
        self._baseline=self._snapshot();self._scheduler=ReleaseScheduler(delay)
        self._step=0;self._seen=set();self._closed=False;self._alerts=[];self._failed=False
    @property
    def assignment_seed(self):return self._assignment_seed
    def _snapshot(self):return hashlib.sha256(canonical_bytes(self._fingerprints())).hexdigest()
    def _check(self):
        if self._snapshot()!=self._baseline:raise FrozenStateError('Frozen model/preprocessing/reference/detector parameters changed')
    def step(self,batch,packets):
        if type(batch) is not MonitorBatch:raise TypeError('Only whitelisted monitoring input accepted')
        if self._closed or batch.step!=self._step+1:raise ValueError('Nonsequential or closed replay')
        if len(batch.windows)!=self._batch_size or (self._max_steps is not None and batch.step>self._max_steps):raise ValueError('Batch differs from locked replay dimensions')
        ids={w.unit_id for w in batch.windows};packets=tuple(packets)
        if any(type(p) is not OutcomePacket for p in packets):raise TypeError('Only immutable endpoint packets accepted')
        if self._expected_units is not None and ids!=self._expected_units[batch.step-1]:raise ValueError('Batch membership differs from locked assignment')
        if self._seen.intersection(ids):raise ValueError('Record reused across batches')
        if len(packets)!=len(ids) or {p.unit_id for p in packets}!=ids:raise ValueError('Outcome/window mismatch')
        lengths={w.unit_id:w.observed_rows for w in batch.windows}
        if any(len(p.endpoints)!=lengths[p.unit_id] for p in packets):raise ValueError('Prediction-window/outcome mismatch')
        try:
            self._check()
            signal=self._monitor(batch) # labels never passed to this callback
            signal_json=json.dumps(signal,sort_keys=True,allow_nan=False)
            input_digest=hashlib.sha256(batch.to_json().encode()).hexdigest()
            self._check();self._scheduler.stage(batch.step,packets)
            released=self._scheduler._release(batch.step,_AFTER_MONITOR)
            self._seen.update(ids);self._step=batch.step
            for value in released:self._review(value)
            self._check()
            self._alerts.append(AlertSnapshot(batch.step,self._assignment_seed,signal_json,input_digest))
            return StepResult(batch.step,signal_json,released,self._assignment_seed,input_digest)
        except BaseException:
            self._closed=True;self._failed=True
            raise
    def flush(self):
        if self._failed:raise FrozenStateError('Failed replay cannot be published')
        if self._closed:return ()
        self._closed=True;released=[]
        try:
            while self._scheduler.pending:
                self._step+=1;self._check()
                values=self._scheduler._release(self._step,_AFTER_MONITOR)
                for value in values:self._review(value)
                self._check();released.extend(values)
        except BaseException:
            self._failed=True
            raise
        return tuple(released) # release-only steps create no monitoring opportunity

    def seal(self):
        if self._failed:raise FrozenStateError('Failed replay cannot be sealed')
        if self._max_steps is not None and len(self._alerts)!=self._max_steps:raise ValueError('Incomplete locked assignment cannot be sealed')
        self.flush();self._check()
        value=[(a.step,a.assignment_seed,a.signal_json,a.monitor_input_sha256) for a in self._alerts]
        digest=hashlib.sha256(canonical_bytes(value)).hexdigest()
        return SealedReplay(self._assignment_seed,tuple(self._alerts),digest,_SEAL_PROOF,self._protocol_sha256)
