"""Distinct covariate and outcome readers. All final PSV access stays disabled."""
from pathlib import Path
from itertools import islice
from .features import COLUMNS,prefix_features
from .partitions import AccessDenied
from .endpoints import OutcomeSeries,adjudicate

HEADER=(*COLUMNS,'SepsisLabel')

def _rows(path):
    with Path(path).open() as f:
        if tuple(f.readline().rstrip('\r\n').split('|'))!=HEADER:raise ValueError('Unexpected schema')
        for line in f:
            fields=line.rstrip('\r\n').split('|')
            if len(fields)!=41:raise ValueError('Unexpected row width')
            yield fields

class CovariateReader:
    __slots__=('_root','_registry')
    def __init__(self,raw_root,registry,*,allow_synthetic=False):
        if not registry.verified and not allow_synthetic:raise AccessDenied('Verified membership registry required')
        if allow_synthetic and Path(raw_root).resolve()==(Path(__file__).resolve().parents[1]/'data/raw').resolve():raise AccessDenied('Synthetic registries cannot read the real source directory')
        self._root=Path(raw_root);self._registry=registry
    def observations(self,key,purpose,direction,heldout_fold=None):
        self._registry.authorize(key,purpose,direction,heldout_fold)
        path=self._root/key.hospital/key.filename
        # The outcome column is discarded without numeric parsing or exposure.
        for fields in islice(_rows(path),24):yield tuple(float(v) for v in fields[:40])
    def prediction_point(self,key,row,purpose,direction,heldout_fold=None):
        if row not in range(1,25):raise ValueError('Outside prediction window')
        prefix=tuple(islice(self.observations(key,purpose,direction,heldout_fold),row))
        if len(prefix)!=row:raise ValueError('Requested hour not observed')
        return prefix_features(prefix)

class AdjudicationVault:
    __slots__=('_root','_registry')
    def __init__(self,raw_root,registry,*,allow_synthetic=False):
        if not registry.verified and not allow_synthetic:raise AccessDenied('Verified membership registry required')
        if allow_synthetic and Path(raw_root).resolve()==(Path(__file__).resolve().parents[1]/'data/raw').resolve():raise AccessDenied('Synthetic registries cannot read the real source directory')
        self._root=Path(raw_root);self._registry=registry
    def adjudicate(self,key,purpose,direction,heldout_fold=None):
        self._registry.authorize(key,purpose,direction,heldout_fold)
        times=[];labels=[]
        for fields in _rows(self._root/key.hospital/key.filename):
            times.append(float(fields[39]));label=float(fields[40])
            if label not in (0,1):raise ValueError('Invalid outcome')
            labels.append(int(label))
        return adjudicate(OutcomeSeries(tuple(times),tuple(labels)))
