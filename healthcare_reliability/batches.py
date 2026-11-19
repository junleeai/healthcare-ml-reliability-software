"""Metadata-only artificial assignments; no outcomes or patient PSV reads."""
from dataclasses import dataclass
import hashlib
from .partitions import Direction,RecordKey

@dataclass(frozen=True,slots=True)
class BatchPlan:
    version:str
    assignment_seed:int
    direction:Direction
    transfer:bool
    size:int
    batches:tuple[tuple[RecordKey,...],...]
    baseline_records:int=3000
    post_records:int=5000
    def __post_init__(self):
        flat=[k for b in self.batches for k in b]
        if self.size not in {250,500,1000} or any(len(b)!=self.size for b in self.batches):raise ValueError('Invalid batch size')
        if len(flat)!=8000 or len(set(flat))!=len(flat):raise ValueError('Within-assignment record reuse or wrong budget')
    @property
    def first_changed_step(self):return self.baseline_records//self.size+1 if self.transfer else None
    @property
    def sensitivity_simulation(self):return self.assignment_seed!=20261118

def make_plan(registry,size=500,seed=20261118,direction=Direction.PRIMARY,transfer=True):
    if not registry.verified:raise ValueError('Verified membership registry required')
    if size not in {250,500,1000} or not isinstance(direction,Direction):raise ValueError('Invalid assignment')
    source=direction.source;target='B' if source=='A' else 'A'
    def ranked(h):
        keys=[k for k in registry.keys() if k.hospital==h and registry.original_role(k)=='final_evaluation']
        return sorted(keys,key=lambda k:hashlib.sha256(f'1.0.0|{seed}|{k.hospital}|{k.filename}'.encode()).hexdigest())
    s=ranked(source);t=ranked(target)
    if len(s)<8000 or len(t)<8000:raise ValueError('Insufficient reserved records')
    chosen=s[:3000]+t[:5000] if transfer else s[:8000]
    return BatchPlan('1.0.0',seed,direction,transfer,size,tuple(tuple(chosen[j:j+size]) for j in range(0,8000,size)))
