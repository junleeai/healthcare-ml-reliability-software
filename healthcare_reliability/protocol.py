"""Content-addressed core protocol lock; no final-data unlock switch."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib,json

PROTOCOL_SHA256='6de922c42f966829ec4af523fa6a5c28a04b07eb00285d54e91b055b80964d00'

def _freeze(value):
    if isinstance(value,dict):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,list):return tuple(_freeze(v) for v in value)
    return value

@dataclass(frozen=True,slots=True)
class ProtocolLock:
    version:str
    sha256:str
    config:object
    @classmethod
    def load(cls,path,expected_sha256=PROTOCOL_SHA256):
        content=Path(path).read_bytes();digest=hashlib.sha256(content).hexdigest()
        if digest!=expected_sha256:raise ValueError('Unreviewed protocol configuration change')
        c=json.loads(content)
        if c['version']!='1.0.0' or c['primary']['binary_primary'] or c['primary']['estimand']!='patient_record_balanced_log_loss_change':raise ValueError('Primary scientific contract changed')
        if c['dataset']['final_psv_access_in_phase3'] or c['monitoring']['adaptation_after_release']:raise ValueError('Forbidden phase action')
        if c['target']['horizon_hours']!=6 or c['target']['candidate_first_rows']!=24:raise ValueError('Unreviewed target')
        return cls(c['version'],digest,_freeze(c))

    def lifecycle(self,registry,plan,secret,monitor,review,fingerprints,*,delay=None):
        from .release import ReplayLifecycle
        from .batches import make_plan
        from .monitoring import opaque_id
        m=self.config['monitoring']
        if not registry.verified or registry.base_digest!=self.config['dataset']['base_manifest_sha256']:raise ValueError('Wrong base membership')
        if plan.version!=self.version or plan.size not in (m['primary_batch_size'],*m['batch_size_sensitivities']):raise ValueError('Unreviewed batch plan')
        if plan.assignment_seed not in (m['assignment_seed'],*m['repeated_assignment_seeds']):raise ValueError('Unreviewed assignment seed')
        if plan!=make_plan(registry,plan.size,plan.assignment_seed,plan.direction,plan.transfer):raise ValueError('Assignment differs from reproducible reserved membership')
        delay=m['delay_steps'] if delay is None else delay
        if delay not in (m['delay_steps'],*m['delay_sensitivities']):raise ValueError('Unreviewed label delay')
        keys={k for b in plan.batches for k in b}
        unit_ids={k:opaque_id(k,secret) for k in keys}
        if len(set(unit_ids.values()))!=len(keys):raise ValueError('Opaque ID collision')
        expected=tuple(frozenset(unit_ids[k] for k in b) for b in plan.batches)
        return ReplayLifecycle(monitor,review,lambda:{'protocol':self.sha256,'fitted_components':fingerprints()},delay,batch_size=plan.size,max_steps=len(plan.batches),expected_units=expected,assignment_seed=plan.assignment_seed,protocol_sha256=self.sha256)

    def loss_comparison(self,reference,current,*,mode,seed,seals):
        from .evaluation import compare
        from .release import SealedReplay
        if not seals or any(type(s) is not SealedReplay or s.protocol_sha256!=self.sha256 for s in seals):raise PermissionError('Primary comparison requires successfully sealed replay logs')
        return compare(reference,current,mode=mode,draws=self.config['statistics']['bootstrap_draws'],seed=seed)
