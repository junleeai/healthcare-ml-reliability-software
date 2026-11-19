"""Phase 5 source-only covariate access and frozen model verification."""
from contextlib import contextmanager
from pathlib import Path
from itertools import islice
import hashlib
import json
import numpy as np
import joblib
from .partitions import RecordKey, Purpose, Direction, AccessDenied
from .model_development import FEATURE_NAMES, membership_fingerprint
from .features import prefix_features
from .access import CovariateReader, AdjudicationVault
from .monitoring import compile_window, opaque_id


def ranked(keys, salt):
    return tuple(sorted(keys,key=lambda k:hashlib.sha256(f'{salt}|{k.hospital}|{k.filename}'.encode()).hexdigest()))


def cohort_roles(registry):
    if not registry.verified:raise AccessDenied('Verified registry required')
    ref=ranked([k for k in registry.keys() if k.hospital=='A' and registry.original_role(k)=='detector_reference'],'phase5-reference-v1')
    check=ranked([k for k in registry.keys() if k.hospital=='A' and registry.original_role(k)=='detector_null'],'phase5-check-v1')
    op=tuple(k for k in registry.keys() if k.hospital=='A' and registry.original_role(k)=='operating_point')
    return {'verification':op,'reference':ref,'calibration':check[:len(check)//2],'heldout':check[len(check)//2:]}


@contextmanager
def phase5_access(raw_root, registry):
    """Tripwire for accidental use. No label-vault calls or raw mutation allowed."""
    roles=cohort_roles(registry);allowed={s:set(keys) for s,keys in roles.items()}
    root=Path(raw_root).resolve();original=Path.open;vault=AdjudicationVault.adjudicate
    audit={'stage':'verification','threshold_sealed':False,'opened':set(),'blocked':0,'vault_calls':0}
    def guarded(path,*args,**kwargs):
        p=path.resolve()
        if p.suffix=='.psv' and root in p.parents:
            rel=p.relative_to(root);key=RecordKey(*rel.parts) if len(rel.parts)==2 else None
            mode=args[0] if args else kwargs.get('mode','r')
            if (key not in allowed.get(audit['stage'],set()) or any(c in mode for c in 'wax+')
                    or (audit['stage']=='heldout' and not audit['threshold_sealed'])):
                audit['blocked']+=1;raise AccessDenied('Forbidden Phase 5 raw role/stage/write')
            audit['opened'].add(key)
        return original(path,*args,**kwargs)
    def denied_vault(*args,**kwargs):
        audit['vault_calls']+=1;raise AccessDenied('Phase 5 has no outcome-vault capability')
    Path.open=guarded;AdjudicationVault.adjudicate=denied_vault
    try:yield audit
    finally:Path.open=original;AdjudicationVault.adjudicate=vault


@contextmanager
def forbid_predictive_fitting():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from .model_development import FrozenPreprocessor
    old=(HistGradientBoostingClassifier.fit,LogisticRegression.fit,FrozenPreprocessor.__dict__['learn'])
    def deny(*args,**kwargs):raise AccessDenied('Predictive fitting/recalibration not authorized in Phase 5')
    HistGradientBoostingClassifier.fit=deny;LogisticRegression.fit=deny;FrozenPreprocessor.learn=classmethod(deny)
    try:yield
    finally:
        HistGradientBoostingClassifier.fit,LogisticRegression.fit,FrozenPreprocessor.learn=old


def load_model(root, config):
    root=Path(root);meta=json.loads((root/'results/phase4/run-v1/artifact-metadata.json').read_text())
    decision=json.loads((root/'results/phase4/run-v1/source-selection.json').read_text())
    if (decision['source']!='A' or decision['B_opened'] or decision['selected']!='boosted_leaves7'
            or meta['selected']!='boosted_leaves7' or tuple(meta['feature_names'])!=FEATURE_NAMES or len(FEATURE_NAMES)!=59):
        raise ValueError('Model selection or feature contract discrepancy')
    path=root/meta['models']['boosted']['file']
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=config['model_sha256'] or sha!=meta['models']['boosted']['sha256']:raise ValueError('Model artifact checksum discrepancy')
    if hashlib.sha256((root/'results/phase4/run-v1/source-selection.json').read_bytes()).hexdigest()!=meta['selection_sha256']:
        raise ValueError('Selection certificate changed')
    model=joblib.load(path)
    if (model.candidate!='boosted_leaves7' or model.excluded_folds or tuple(model.preprocessor.medians)!=tuple(meta['models']['boosted']['medians'])
            or tuple(model.preprocessor.scales)!=tuple(meta['models']['boosted']['scales'])
            or model.estimator.get_params()!=meta['models']['boosted']['parameters']):
        raise ValueError('Loaded model/preprocessor contract discrepancy')
    return model,meta


def verify_phase4_predictions(root,registry,model):
    """Use only saved alignment group/row/keys and A operating covariates.

    The npz outcome field is never loaded. This alignment is verification-only;
    it never selects monitoring windows or calibrates a detector.
    """
    root=Path(root);reader=CovariateReader(root/'data/raw',registry)
    with np.load(root/'data/phase4/run-v1/A_operating-alignment.npz') as a:
        groups,rows,keys=a['group'],a['rows'],a['keys']
    with np.load(root/'data/phase4/run-v1/development-predictions.npz') as p:expected=p['A_operating_boosted']
    prediction=np.empty(len(rows))
    for g,text in enumerate(keys):
        key=RecordKey(*str(text).split('/'))
        registry.authorize(key,Purpose.OPERATING,Direction.PRIMARY)
        obs=tuple(reader.observations(key,Purpose.OPERATING,Direction.PRIMARY))
        points=np.array([prefix_features(obs[:j]).values for j in range(1,len(obs)+1)])
        prob=model.predict_matrix(points)
        mask=groups==g;prediction[mask]=prob[rows[mask]-1]
    if not np.array_equal(prediction,expected):raise ValueError('Saved model failed Phase 4 bitwise probability reproduction')
    return {'prediction_hours_checked':len(expected),'operating_records_checked':len(keys),'bitwise_equal':True,
            'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),
            'saved_outcome_arrays_loaded':False,'runtime_pipeline_fingerprint':model.fingerprint()}


def compiled_prediction_window(unit_id, observations, model):
    """Batch only independent causal prefix rows, then use the Phase 3 compiler."""
    obs=tuple(islice(observations,24))
    if not obs:raise ValueError('Empty window')
    x=np.array([prefix_features(obs[:j]).values for j in range(1,len(obs)+1)])
    z=model.preprocessor.transform(x);prob=model.predict_transformed_matrix(z)
    def cached(point):
        if not np.array_equal(np.asarray(point.values),z[point.row-1]):raise ValueError('Causal prefix cache mismatch')
        return float(prob[point.row-1])
    return compile_window(unit_id,iter(obs),model.transform_point,cached)


def collect_windows(root,registry,keys,stage,model,secret):
    roles=cohort_roles(registry)
    if stage not in ('reference','calibration','heldout') or not set(keys)<=set(roles[stage]):
        raise AccessDenied('Reference/check membership mixing')
    reader=CovariateReader(Path(root)/'data/raw',registry)
    purpose=Purpose.REFERENCE if stage=='reference' else Purpose.CHECK
    windows=[]
    for key in keys:
        registry.authorize(key,purpose,Direction.PRIMARY)
        w=compiled_prediction_window(opaque_id(key,secret),reader.observations(key,purpose,Direction.PRIMARY),model)
        windows.append(w.monitor)
    return tuple(windows)
