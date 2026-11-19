"""Invented software contracts; no fitting, data, reference or research results."""
import json
import numpy as np
import pytest
from scipy.stats import ks_2samp
from healthcare_reliability.features import prefix_features
from healthcare_reliability.endpoints import OutcomeSeries,Eligibility,Endpoint,adjudicate
from healthcare_reliability.monitoring import MonitorRecord,MonitorBatch
from healthcare_reliability.release import ReplayLifecycle,OutcomePacket,FrozenStateError
from healthcare_reliability.partitions import RecordKey,PartitionRegistry,Purpose,AccessDenied
from healthcare_reliability.evaluation import RecordLoss,estimate,compare
from healthcare_reliability.detectors import weighted_ks

def observations(n=12):
    x=np.full((n,40),np.nan);x[:,0]=np.arange(n)+70;x[:,35]=0;x[:,39]=np.arange(1,n+1)
    return x

def window(unit='invented'):
    return MonitorRecord(unit,(0.,)*40,(0.,)*19,.2,1)

def loss(i,value,hours=1):
    return RecordLoss(RecordKey('A',f'p{i}.psv'),value*hours,hours,0,'invented-digest')

def test_future_observations_cannot_change_selected_prefix():
    x=observations();before=prefix_features(x[:4]);x[4:]=9999
    np.testing.assert_allclose(before.values,prefix_features(x[:4]).values,equal_nan=True)

def test_missing_and_observed_zero_have_distinct_coverage():
    x=observations(2);missing=prefix_features(x).values
    x[:,0]=0;zero=prefix_features(x).values
    assert missing[40]==zero[40]==1.
    x[:,0]=np.nan;assert prefix_features(x).values[40]==0.
    assert zero[0]==0.

@pytest.mark.parametrize('positive',[None,1,4])
def test_endpoint_censoring_and_horizon(positive):
    labels=tuple(int(positive is not None and j>=positive) for j in range(1,19))
    a=adjudicate(OutcomeSeries(tuple(range(1,19)),labels))
    if positive==1:assert all(e.state is Eligibility.LEFT_CENSORED for e in a.endpoints)
    elif positive is None:
        assert a.endpoints[11].eligible and not a.endpoints[12].eligible
    else:
        assert a.effective_onset==10 and a.endpoints[3].outcome==1
        assert a.endpoints[9].state is Eligibility.POST_ONSET

@pytest.mark.parametrize('name',['outcome','SepsisLabel','onset','eligible','future'])
def test_label_free_schema_rejects_extra_fields(name):
    text=json.loads(MonitorBatch(1,(window(),)).to_json())
    text['windows'][0][name]=1
    with pytest.raises(ValueError):MonitorBatch.from_json(json.dumps(text))

def test_duplicate_records_rejected():
    with pytest.raises(ValueError):MonitorBatch(1,(window(),window()))

@pytest.mark.parametrize('a,b',[(np.array([0.,1.,2.]),np.array([0.,1.,2.])),(np.array([0.,0.,1.]),np.array([1.,2.,2.]))])
def test_empirical_ks_arithmetic(a,b):
    got=weighted_ks(a,b,np.ones((1,len(a))),np.ones((1,len(b))))[0]
    expected=ks_2samp(a,b).statistic*np.sqrt(len(a)*len(b)/(len(a)+len(b)))
    assert got==pytest.approx(expected)

def test_record_and_hour_weighting_are_distinct():
    e=estimate([loss(1,.1,1),loss(2,.9,9),loss(3,0.,0)])
    assert e.scored_records==2 and e.patient_balanced==pytest.approx(.5)
    assert e.hour_weighted==pytest.approx(.82)

def test_paired_cluster_multiplicity_retains_constant_difference():
    r=[loss(i,i/10,i) for i in range(1,5)];c=[loss(i,i/10+.2,i) for i in range(1,5)]
    d=compare(r,c,mode='paired',seed=4,draws=100)
    assert d.patient_interval.lower==pytest.approx(.2)
    assert d.patient_interval.upper==pytest.approx(.2)

def test_independent_mode_rejects_overlap():
    with pytest.raises(ValueError):compare([loss(1,.1)],[loss(1,.2)],mode='independent',draws=50)

@pytest.mark.parametrize('delay',[0,3,6])
def test_monitor_runs_before_artificial_release(delay):
    events=[]
    def monitor(batch):events.append('monitor');return {'signal':False}
    def review(packet):events.append('review')
    run=ReplayLifecycle(monitor,review,lambda:{'fixed':'invented'},delay,batch_size=1)
    packet=OutcomePacket('invented',(Endpoint(1,Eligibility.NEGATIVE,0),))
    run.step(MonitorBatch(1,(window(),)),(packet,));run.seal()
    assert events==['monitor','review']

def test_state_changes_fail_closed():
    state={'fixed':1}
    run=ReplayLifecycle(lambda b:{},lambda p:None,lambda:state,0,batch_size=1)
    state['fixed']=2
    with pytest.raises(FrozenStateError):run.step(MonitorBatch(1,(window(),)),(OutcomePacket('invented',(Endpoint(1,Eligibility.NEGATIVE,0),)),))

def test_final_role_denial_does_not_need_manifest_contents():
    key=RecordKey('A','p1.psv');registry=PartitionRegistry({key:'final_evaluation'},'invented')
    with pytest.raises(AccessDenied):registry.authorize(key,Purpose.AUDIT)
