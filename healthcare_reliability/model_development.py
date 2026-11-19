"""Phase 4 development only. No detector, replay, final access, or adaptation.

Fitting entry points validate registered record groups before learned operations.
The frozen candidate exposes prediction, never a preprocessing refit operation.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .access import CovariateReader, AdjudicationVault
from .features import PredictorInput, prefix_features, CLINICAL
from .partitions import (POOL, AccessDenied, Purpose, Direction, RecordKey,
                         canonical_bytes)

_SEAL_PROOF = object()
SEED = 20261117
CLIP = 1e-6
CANDIDATES = ('constant', 'logistic_C0.1', 'logistic_C1', 'logistic_C10',
              'boosted_leaves7', 'boosted_leaves15')
FEATURE_NAMES = tuple([f'{s}_last6' for s in CLINICAL] +
                      [f'{s}_median6' for s in CLINICAL] + ['Gender', 'ICULOS'] +
                      [f'{s}_coverage6' for s in CLINICAL])


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def membership_fingerprint(keys):
    return digest([f'{k.hospital}/{k.filename}' for k in sorted(keys)])


def require_verified(registry, allow_synthetic=False):
    if not registry.verified and not allow_synthetic:
        raise AccessDenied('Pinned verified membership required')


def development_keys(registry, stage):
    """Explicitly narrow the broad Phase 3 AUDIT capability."""
    if stage == 'source':
        return tuple(k for k in registry.keys() if k.hospital == 'A' and registry.original_role(k) in POOL)
    if stage == 'operating':
        return tuple(k for k in registry.keys() if k.hospital == 'A' and registry.original_role(k) == 'operating_point')
    if stage == 'transfer':
        return tuple(k for k in registry.keys() if k.hospital == 'B' and registry.original_role(k) in POOL)
    raise AccessDenied('Unknown Phase 4 stage')


@dataclass(frozen=True)
class SelectionSeal:
    """Digest-verified saved source decision, required before full-source refits."""
    payload: dict
    sha256: str
    _proof: object = field(default=None, repr=False)

    @property
    def valid(self):
        return self._proof is _SEAL_PROOF and digest(self.payload) == self.sha256

    @classmethod
    def load(cls, path, expected_digest, registry):
        require_verified(registry)
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise AccessDenied('Source decision seal changed')
        p = json.loads(raw)
        if (p.get('source') != 'A' or p.get('base_manifest_sha256') != registry.base_digest
                or p.get('training_membership_sha256') != membership_fingerprint(development_keys(registry, 'source'))
                or p.get('selected') not in CANDIDATES
                or not p.get('B_opened') is False):
            raise AccessDenied('Not a valid A-only selection decision')
        return cls(p, expected_digest, _SEAL_PROOF)


def authorize_fit(registry, keys, excluded_folds, *, seal=None, allow_synthetic=False):
    require_verified(registry, allow_synthetic)
    keys = tuple(keys)
    if not keys or len(set(keys)) != len(keys):
        raise AccessDenied('Nonempty distinct fitting records required')
    excluded = frozenset(excluded_folds)
    if excluded and (not excluded <= set(range(1, 6)) or len(excluded) not in (1, 2)):
        raise AccessDenied('Invalid grouped validation split')
    if not excluded:
        if (type(seal) is not SelectionSeal or not seal.valid or not registry.verified
                or membership_fingerprint(keys) != seal.payload['training_membership_sha256']):
            raise AccessDenied('Full-source refit requires sealed selection and exact pool membership')
    for k in keys:
        if k.hospital != 'A' or registry.original_role(k) not in POOL or k.fold in excluded:
            raise AccessDenied('Target hospital, wrong role, or held-out fold in fit')
        if excluded:
            for fold in excluded:
                registry.authorize(k, Purpose.TRAIN, Direction.PRIMARY, fold)


@contextmanager
def raw_access_tripwire(raw_root, registry):
    """Fail before any forbidden raw file opens; collect actual read membership.

    Single-process audit aid, not an OS security boundary. Stages are explicit and
    transfer cannot start before the caller supplies a verified source seal.
    """
    root = Path(raw_root).resolve()
    audit = {'stage': 'source', 'seal': None, 'opened': set(), 'attempts_blocked': 0}
    original = Path.open
    allowed_by_stage = {s: set(development_keys(registry, s)) for s in ('source', 'operating', 'transfer')}
    def guarded(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved.suffix == '.psv' and root in resolved.parents:
            relative = resolved.relative_to(root)
            key = RecordKey(*relative.parts) if len(relative.parts) == 2 else None
            allowed = allowed_by_stage[audit['stage']]
            if key not in allowed or (audit['stage'] != 'source' and (type(audit['seal']) is not SelectionSeal or not audit['seal'].valid)):
                audit['attempts_blocked'] += 1
                raise AccessDenied('Phase 4 raw-access tripwire: forbidden role/stage')
            mode = args[0] if args else kwargs.get('mode', 'r')
            if any(c in mode for c in 'wax+'):
                raise AccessDenied('Raw mutation forbidden')
            audit['opened'].add(key)
        return original(path, *args, **kwargs)
    Path.open = guarded
    try:
        yield audit
    finally:
        Path.open = original


@dataclass
class DevelopmentData:
    keys: tuple
    group: np.ndarray
    rows: np.ndarray
    x: np.ndarray
    y: np.ndarray
    counts: np.ndarray
    folds: np.ndarray

    def __post_init__(self):
        n = len(self.y)
        if (not self.keys or len(set(self.keys)) != len(self.keys) or self.x.shape != (n,59)
                or self.group.shape != (n,) or self.rows.shape != (n,)
                or self.counts.shape != (len(self.keys),) or self.folds.shape != (len(self.keys),)
                or np.any((self.group < 0) | (self.group >= len(self.keys)))
                or not np.isin(self.y,[0,1]).all() or not np.isin(self.rows,range(1,25)).all()
                or not np.array_equal(self.counts,np.bincount(self.group,minlength=len(self.keys)))
                or not np.array_equal(self.folds,[k.fold for k in self.keys])
                or len(set(zip(self.group,self.rows))) != n):
            raise ValueError('Invalid record-hour alignment')

    @property
    def weights(self):
        return 1.0 / self.counts[self.group]

    def mask_excluding(self, excluded):
        return ~np.isin(self.folds[self.group], list(excluded))


def load_development(raw_root, registry, stage, *, seal=None):
    require_verified(registry)
    if stage != 'source' and (type(seal) is not SelectionSeal or not seal.valid):
        raise AccessDenied('Seal A decision before opening operating or transfer records')
    keys = development_keys(registry, stage)
    reader, vault = CovariateReader(raw_root, registry), AdjudicationVault(raw_root, registry)
    xx, yy, gg, rr, counts = [], [], [], [], []
    for g, key in enumerate(keys):
        purpose = Purpose.OPERATING if stage == 'operating' else Purpose.AUDIT
        observations = tuple(reader.observations(key, purpose, Direction.PRIMARY))
        endpoints = vault.adjudicate(key, purpose, Direction.PRIMARY).endpoints
        n = 0
        # Feature construction sees covariates only, never an endpoint value.
        points = tuple(prefix_features(observations[:j]) for j in range(1, len(observations) + 1))
        for point, endpoint in zip(points, endpoints, strict=True):
            if endpoint.eligible:
                xx.append(point.values); yy.append(endpoint.outcome)
                gg.append(g); rr.append(point.row); n += 1
        counts.append(n)
    return DevelopmentData(keys, np.asarray(gg, dtype=int), np.asarray(rr, dtype=int),
                           np.asarray(xx), np.asarray(yy, dtype=int), np.asarray(counts),
                           np.asarray([k.fold for k in keys]))


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    if len(v) == 0 or w.sum() <= 0:
        raise ValueError('No observed training support')
    cumulative = np.cumsum(w, dtype=np.longdouble)
    target = q * cumulative[-1] - 8*np.finfo(float).eps*cumulative[-1]
    return float(v[min(np.searchsorted(cumulative, target, side='left'), len(v)-1)])


@dataclass(frozen=True)
class FrozenPreprocessor:
    medians: tuple
    scales: tuple

    @classmethod
    def learn(cls, x, weights):
        if x.ndim != 2 or x.shape[1] != 59 or not np.isfinite(x[:, 38:40]).all():
            raise ValueError('Require observed Gender and ICULOS and 59 raw coordinates')
        med, scale = [], []
        for j in range(40):
            valid = np.isfinite(x[:, j])
            if not valid.any():
                # Explicit training-only fallback for a wholly unobserved clinical coordinate.
                m, s = 0., 1.
            else:
                m = weighted_quantile(x[valid, j], weights[valid], .5)
                s = weighted_quantile(x[valid, j], weights[valid], .75) - weighted_quantile(x[valid, j], weights[valid], .25)
                if s <= 0: s = 1.
            med.append(m); scale.append(s)
        return cls(tuple(med), tuple(scale))

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 59 or np.isinf(x).any() or not np.isfinite(x[:, 38:40]).all():
            raise ValueError('Out-of-contract prediction features')
        if not np.isfinite(x[:, 40:]).all() or np.any((x[:, 40:] < 0) | (x[:, 40:] > 1)):
            raise ValueError('Invalid coverage')
        out = x.copy()
        out[:, :40] = (np.where(np.isfinite(x[:, :40]), x[:, :40], self.medians) - self.medians) / self.scales
        if not np.isfinite(out).all(): raise ValueError('Nonfinite transformed input')
        return out

    def point(self, point):
        if type(point) is not PredictorInput: raise TypeError('PredictorInput required')
        return PredictorInput(point.row, tuple(self.transform([point.values])[0]))


@dataclass(frozen=True)
class FrozenCandidate:
    candidate: str
    preprocessor: FrozenPreprocessor
    estimator: object
    training_membership_sha256: str
    excluded_folds: tuple

    def predict_matrix(self, raw_features):
        return self.predict_transformed_matrix(self.preprocessor.transform(raw_features))

    def predict_transformed_matrix(self, transformed):
        if self.candidate == 'constant':
            p = np.full(len(transformed), self.estimator)
        else:
            with threadpool_limits(limits=1):
                p = self.estimator.predict_proba(transformed)[:, 1]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid model probability')
        return p

    def transform_point(self, point):
        return self.preprocessor.point(point)

    def predict_transformed(self, point):
        return float(self.predict_transformed_matrix(np.asarray([point.values]))[0])

    def predict_point(self, point):
        return self.predict_transformed(self.transform_point(point))

    def fingerprint(self):
        import joblib
        return joblib.hash(self, hash_name='sha1')


def fit_candidate(data, registry, candidate, excluded_folds, *, seal=None, allow_synthetic=False):
    mask = data.mask_excluding(excluded_folds)
    fit_keys = tuple(k for k in data.keys if k.fold not in excluded_folds)
    authorize_fit(registry, fit_keys, excluded_folds, seal=seal, allow_synthetic=allow_synthetic)
    if candidate not in CANDIDATES: raise ValueError('Unspecified candidate')
    x, y, w = data.x[mask], data.y[mask], data.weights[mask]
    prep = FrozenPreprocessor.learn(x, w)
    z = prep.transform(x)
    if candidate == 'constant':
        estimator = float(np.average(y, weights=w))
    elif candidate.startswith('logistic'):
        estimator = LogisticRegression(C=float(candidate.split('_C')[1]), solver='newton-cholesky',
                                       max_iter=100, tol=1e-6, random_state=SEED, class_weight=None)
    else:
        estimator = HistGradientBoostingClassifier(max_leaf_nodes=int(candidate.split('leaves')[1]),
            learning_rate=.05, max_iter=100, min_samples_leaf=50, l2_regularization=1.,
            early_stopping=False, random_state=SEED, class_weight=None)
    if candidate != 'constant':
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit(z, y, sample_weight=w)
    return FrozenCandidate(candidate, prep, estimator, membership_fingerprint(fit_keys), tuple(sorted(excluded_folds)))


def record_losses(data, p):
    p = np.asarray(p)
    if p.shape != data.y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid aligned probabilities')
    p = np.clip(p, CLIP, 1-CLIP)
    loss = -(data.y * np.log(p) + (1-data.y) * np.log1p(-p))
    sums = np.bincount(data.group, weights=loss, minlength=len(data.keys))
    means = np.divide(sums, data.counts, out=np.full(len(data.keys), np.nan), where=data.counts>0)
    return loss, sums, means


def balanced_loss(data, p, records=None):
    _, _, means = record_losses(data, p)
    mask = data.counts > 0
    if records is not None: mask &= np.asarray(records)
    return float(means[mask].mean())


def choose(scores):
    best = min(scores.values())
    return next(c for c in CANDIDATES if c in scores and scores[c] <= best + 1e-4)


def family(candidate):
    return candidate.split('_')[0]


def operating_threshold(data, p, registry, *, allow_synthetic=False):
    require_verified(registry, allow_synthetic)
    for key in data.keys:
        registry.authorize(key, Purpose.OPERATING, Direction.PRIMARY)
    neg = data.y == 0
    n = np.bincount(data.group[neg], minlength=len(data.keys))
    if not (n > 0).any(): raise ValueError('No operating nonevents')
    w = 1 / n[data.group[neg]]
    threshold = weighted_quantile(np.asarray(p)[neg], w, .95)
    if np.average(np.asarray(p)[neg] >= threshold, weights=w) > .05 + 1e-14:
        threshold = float(np.nextafter(threshold, np.inf))
    return threshold


def support(data):
    pos = np.bincount(data.group, weights=data.y, minlength=len(data.keys)).astype(int)
    scored = data.counts > 0
    return dict(records=len(data.keys), scored_records=int(scored.sum()),
                unscored_records=int((~scored).sum()), positive_opportunity_records=int((pos>0).sum()),
                negative_only_records=int((scored & (pos==0)).sum()), eligible_hours=len(data.y),
                positive_hours=int(data.y.sum()), negative_hours=int((data.y==0).sum()),
                hour_prevalence=float(data.y.mean()),
                patient_balanced_hour_prevalence=float(np.sum(data.weights*data.y)/scored.sum()))


def rank_statistics(y, p, w):
    """Tie-correct weighted AUROC and AP; AP baseline is weighted prevalence."""
    order = np.argsort(-p, kind='stable')
    yy, pp, ww = y[order], p[order], w[order]
    ends = np.r_[np.flatnonzero(pp[:-1] != pp[1:]), len(pp)-1]
    tp = np.cumsum(ww*yy)[ends]; fp = np.cumsum(ww*(1-yy))[ends]
    if tp[-1] <= 0 or fp[-1] <= 0: return None, None
    auc = np.trapezoid(np.r_[0,tp/tp[-1]], np.r_[0,fp/fp[-1]])
    ap = np.sum(np.diff(np.r_[0,tp/tp[-1]]) * tp/(tp+fp))
    return float(auc), float(ap)


def threshold_stats(data, p, threshold, multiplicity=None):
    m = np.ones(len(data.keys)) if multiplicity is None else multiplicity
    neg, pos = data.y==0, data.y==1
    pred = p >= threshold
    result = {}
    for label, mask in [('sensitivity', pos), ('false_positive_rate', neg)]:
        counts = np.bincount(data.group[mask], minlength=len(data.keys))
        hit = np.bincount(data.group[mask], weights=pred[mask], minlength=len(data.keys))
        good = counts > 0
        result[label] = float(np.average(hit[good]/counts[good], weights=m[good])) if m[good].sum() else None
    result['specificity'] = None if result['false_positive_rate'] is None else 1-result['false_positive_rate']
    return result


def calibration(data, p):
    w, y = data.weights, data.y
    z = logit(np.clip(p, CLIP, 1-CLIP))
    mean_gap = float(np.average(p-y, weights=w))
    try:
        intercept = float(brentq(lambda a: np.dot(w, expit(a+z)-y), -30, 30))
    except ValueError: intercept = None
    result = {'mean_probability_minus_prevalence': mean_gap, 'offset_intercept': intercept,
              'joint_intercept': None, 'slope': None, 'slope_status': 'insufficient_event_records_or_constant_predictions'}
    if support(data)['positive_opportunity_records'] >= 100 and np.std(z) > 1e-8:
        def obj(beta):
            eta = beta[0]+beta[1]*z
            residual = w*(expit(eta)-y)
            return np.sum(w*(np.logaddexp(0,eta)-y*eta)), np.array([residual.sum(), np.dot(residual,z)])
        fit = minimize(obj, [0.,1.], jac=True, method='BFGS', options={'gtol':1e-5, 'maxiter':200})
        if np.isfinite(fit.x).all() and (fit.success or np.max(np.abs(fit.jac)) < 1e-4):
            result.update(joint_intercept=float(fit.x[0]), slope=float(fit.x[1]), slope_status='diagnostic_only_not_applied')
    return result


def reliability(data, p):
    edges = np.array([0,.005,.01,.02,.05,.1,.2,.5,1.])
    bins = np.minimum(np.searchsorted(edges, p, side='right')-1,len(edges)-2)
    rows=[]
    for j in range(len(edges)-1):
        mask=bins==j
        if not mask.any(): continue
        w=data.weights[mask]
        rows.append(dict(lower=float(edges[j]),upper=float(edges[j+1]),hours=int(mask.sum()),
            records=len(np.unique(data.group[mask])), positive_records=len(np.unique(data.group[mask & (data.y==1)])),
            weight=float(w.sum()),mean_prediction=float(np.average(p[mask],weights=w)),
            observed_fraction=float(np.average(data.y[mask],weights=w))))
    return rows


def evaluate(data, p, threshold, *, seed=SEED, loss_draws=2000, secondary_draws=500):
    p=np.asarray(p)
    loss,sums,means=record_losses(data,p)
    ids=np.flatnonzero(data.counts>0); w=data.weights
    auc,ap=rank_statistics(data.y,p,w)
    hour_auc,hour_ap=rank_statistics(data.y,p,np.ones(len(p)))
    result={'support':support(data),'patient_balanced_log_loss':float(means[ids].mean()),
            'hour_log_loss':float(loss.mean()),'weighted_auroc':auc,'weighted_average_precision':ap,
            'hour_auroc':hour_auc,'hour_average_precision':hour_ap,
            'calibration':calibration(data,p),'reliability':reliability(data,p),
            'threshold':float(threshold),'operating':threshold_stats(data,p,threshold),
            'confidence':{'method':'record-cluster percentile bootstrap, fixed fitted predictions; conditional descriptive OOF uncertainty',
                          'loss_draws':loss_draws,'secondary_draws':secondary_draws}}
    rng=np.random.default_rng(seed)
    draws={k:[] for k in ['patient_balanced_log_loss','hour_log_loss','weighted_auroc','weighted_average_precision','sensitivity','false_positive_rate']}
    # Cache the rank order once; bootstrap repeats records including all their hours.
    order=np.argsort(-p,kind='stable'); yy=data.y[order]; pp=p[order]; ww=w[order]; group=data.group[order]
    ends=np.r_[np.flatnonzero(pp[:-1]!=pp[1:]),len(pp)-1]
    for b in range(loss_draws):
        sample=rng.choice(ids,len(ids),replace=True)
        draws['patient_balanced_log_loss'].append(float(means[sample].mean()))
        draws['hour_log_loss'].append(float(sums[sample].sum()/data.counts[sample].sum()))
        if b>=secondary_draws: continue
        mult=np.bincount(sample,minlength=len(data.keys))
        weight=ww*mult[group]
        tp=np.cumsum(weight*yy)[ends]; fp=np.cumsum(weight*(1-yy))[ends]
        if tp[-1]>0 and fp[-1]>0:
            draws['weighted_auroc'].append(float(np.trapezoid(np.r_[0,tp/tp[-1]],np.r_[0,fp/fp[-1]])))
            denom=tp+fp
            precision=np.divide(tp,denom,out=np.zeros_like(tp),where=denom>0)
            draws['weighted_average_precision'].append(float(np.sum(np.diff(np.r_[0,tp/tp[-1]])*precision)))
        st=threshold_stats(data,p,threshold,mult)
        for k in ('sensitivity','false_positive_rate'):
            if st[k] is not None: draws[k].append(st[k])
    for k,values in draws.items():
        expected=loss_draws if k in ('patient_balanced_log_loss','hour_log_loss') else secondary_draws
        result['confidence'][k]={'interval95':np.quantile(values,[.025,.975]).tolist() if len(values)>=.95*expected else None,
                                'defined_draws':len(values)}
    result['confidence']['specificity']={'interval95': [1-result['confidence']['false_positive_rate']['interval95'][1],1-result['confidence']['false_positive_rate']['interval95'][0]] if result['confidence']['false_positive_rate']['interval95'] else None}
    result['risk_distribution']={'quantiles':np.quantile(p,[0,.01,.1,.5,.9,.99,1]).tolist(),
                                 'fraction_below_0.01':float((p<.01).mean()),'fraction_above_0.5':float((p>.5).mean())}
    coverage=data.x[:,40:].mean(axis=1)
    # Strata are descriptive record-level mean coverage over eligible prefixes.
    record_coverage=np.divide(np.bincount(data.group,weights=coverage,minlength=len(data.keys)),data.counts,
                              out=np.zeros(len(data.keys)),where=data.counts>0)
    rbin=np.minimum(np.searchsorted([0,.1,.25,.5,1],record_coverage,side='right')-1,3)
    pos=np.bincount(data.group,weights=data.y,minlength=len(data.keys))
    result['missingness_strata']=[]
    for j,(lo,hi) in enumerate(zip([0,.1,.25,.5],[.1,.25,.5,1])):
        mask=(rbin==j)&(data.counts>0)
        if mask.any():result['missingness_strata'].append(dict(coverage_lower=lo,coverage_upper=hi,records=int(mask.sum()),
            positive_records=int(((pos>0)&mask).sum()),patient_balanced_log_loss=float(means[mask].mean())))
    return result
