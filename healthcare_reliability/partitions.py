"""Pinned record memberships and source/fold authorization."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
import hashlib,json,re

BASE_MANIFEST_SHA256='332b8d28e7507847054af2ab2a42569abee698339733dcaaa2153f97f2551118'
POOL=frozenset({'model_train','hyperparameter_validation','probability_calibration'})
ROLES=POOL | {'operating_point','detector_reference','detector_null','final_evaluation'}

_VERIFIED_REGISTRY=object()

class AccessDenied(PermissionError):pass
class Purpose(str,Enum):
    TRAIN='train'
    SELECT='select'
    OPERATING='operating'
    REFERENCE='reference'
    CHECK='check'
    AUDIT='development_audit'
    REPLAY='development_replay'
class Direction(str,Enum):
    PRIMARY='A_to_B'
    SECONDARY='B_to_A'
    @property
    def source(self):return 'A' if self is Direction.PRIMARY else 'B'

@dataclass(frozen=True,slots=True,order=True)
class RecordKey:
    hospital:str
    filename:str
    def __post_init__(self):
        if self.hospital not in {'A','B'} or not re.fullmatch(r'p\d+\.psv',self.filename):raise ValueError('Invalid hospital-qualified record key')
    @property
    def fold(self):
        s=f'phase2c-cv-v1|{self.hospital}|{self.filename}'
        return 1+int(hashlib.sha256(s.encode()).hexdigest(),16)%5

def canonical_bytes(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()

def write_once(path,value):
    path=Path(path);data=canonical_bytes(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Immutable artifact differs; use a reviewed new version')
    else:
        with path.open('xb') as f:f.write(data)
    return hashlib.sha256(data).hexdigest()

class PartitionRegistry:
    __slots__=('_roles','base_digest','_verified')
    def __init__(self,roles,base_digest,*,_verification=None):
        if len(roles)!=len(set(roles)) or any(r not in ROLES for r in roles.values()):raise ValueError('Invalid membership')
        self._roles=MappingProxyType(dict(roles));self.base_digest=base_digest;self._verified=(_verification is _VERIFIED_REGISTRY)
    def __setattr__(self,name,value):
        if hasattr(self,name):raise AttributeError('Immutable partition registry')
        object.__setattr__(self,name,value)
    def __delattr__(self,name):raise AttributeError('Immutable partition registry')
    @classmethod
    def load(cls,path,expected_digest=BASE_MANIFEST_SHA256):
        if expected_digest!=BASE_MANIFEST_SHA256:raise ValueError('Unreviewed base membership digest')
        content=Path(path).read_bytes();digest=hashlib.sha256(content).hexdigest()
        if digest!=expected_digest:raise ValueError('Base membership checksum mismatch')
        rows=json.loads(content);roles={RecordKey(r['hospital'],r['filename']):r['role'] for r in rows}
        if len(roles)!=len(rows):raise ValueError('Duplicate record key')
        return cls(roles,digest,_verification=_VERIFIED_REGISTRY)
    @property
    def verified(self):return self._verified
    def keys(self):return tuple(sorted(self._roles))
    def original_role(self,key):return self._roles[key]
    def consolidated_role(self,key):
        role=self.original_role(key)
        return 'training_internal_validation' if role in POOL else role
    def authorize(self,key,purpose,direction=Direction.PRIMARY,heldout_fold=None):
        if not isinstance(purpose,Purpose) or not isinstance(direction,Direction):raise AccessDenied('Explicit purpose and direction required')
        if key not in self._roles:raise AccessDenied('Unregistered record')
        role=self._roles[key]
        # Foundations have no final-data unlock path, even for feature-only requests.
        if role=='final_evaluation':raise AccessDenied('Final PSV access is disabled in Phase 3')
        if purpose in {Purpose.AUDIT,Purpose.REPLAY}:return
        if key.hospital!=direction.source:raise AccessDenied('Evaluation-hospital data forbidden for source development')
        if purpose in {Purpose.TRAIN,Purpose.SELECT}:
            if role not in POOL:raise AccessDenied('Not in approved source training pool')
            if heldout_fold not in range(1,6):raise AccessDenied('Explicit internal held-out fold required')
            if (key.fold==heldout_fold)!=(purpose is Purpose.SELECT):raise AccessDenied('Training/selection fold contamination')
        else:
            expected={Purpose.OPERATING:'operating_point',Purpose.REFERENCE:'detector_reference',Purpose.CHECK:'detector_null'}[purpose]
            if role!=expected:raise AccessDenied('Wrong development role')
    def consolidated_manifest(self):
        return dict(version='1.0.0',base_manifest_sha256=self.base_digest,records=[dict(hospital=k.hospital,filename=k.filename,original_role=self._roles[k],role=self.consolidated_role(k),internal_fold=k.fold if self._roles[k] in POOL else None) for k in self.keys()])
