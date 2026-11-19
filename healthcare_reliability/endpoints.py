"""Authoritative retrospective endpoint specification; never predictor inputs."""
from dataclasses import dataclass
from enum import Enum
import math

class Eligibility(str,Enum):
    POSITIVE='eligible_positive'
    NEGATIVE='eligible_negative'
    LEFT_CENSORED='left_censored'
    POST_ONSET='at_or_after_onset'
    INCOMPLETE='insufficient_followup'

@dataclass(frozen=True,slots=True)
class OutcomeSeries:
    icu_hours:tuple[float,...]
    labels:tuple[int,...]
    def __post_init__(self):
        if not isinstance(self.icu_hours,tuple) or not isinstance(self.labels,tuple):raise TypeError('Immutable time/label tuples required')
        if not self.icu_hours or len(self.icu_hours)!=len(self.labels):raise ValueError('Nonempty aligned outcome sequence required')
        if any(not math.isfinite(t) or t<=0 for t in self.icu_hours):raise ValueError('Invalid ICU time')
        if any(b-a!=1 for a,b in zip(self.icu_hours,self.icu_hours[1:])):raise ValueError('Nonconsecutive ICU time')
        if any(v not in (0,1) for v in self.labels) or any(b<a for a,b in zip(self.labels,self.labels[1:])):raise ValueError('Invalid binary monotone labels')

@dataclass(frozen=True,slots=True)
class Endpoint:
    row:int
    state:Eligibility
    outcome:int|None
    def __post_init__(self):
        if type(self.row) is not int or not 1<=self.row<=24 or not isinstance(self.state,Eligibility):raise ValueError('Invalid endpoint')
        expected=1 if self.state is Eligibility.POSITIVE else 0 if self.state is Eligibility.NEGATIVE else None
        if self.outcome!=expected:raise ValueError('Outcome/state contradiction')
    @property
    def eligible(self):return self.outcome is not None

@dataclass(frozen=True,slots=True)
class Adjudication:
    endpoints:tuple[Endpoint,...]
    effective_onset:float|None
    # Effective onset is evaluator metadata, not a clinical event timestamp.

def adjudicate(series:OutcomeSeries)->Adjudication:
    times,y=series.icu_hours,series.labels
    first_positive=next((j for j,v in enumerate(y) if v==1),None)
    onset=times[first_positive]+6 if y[0]==0 and first_positive is not None else None
    endpoints=[]
    for j,t in enumerate(times[:24]):
        if y[0]==1:state=Eligibility.LEFT_CENSORED
        elif onset is not None and t>=onset:state=Eligibility.POST_ONSET
        elif onset is not None and t<onset<=t+6 and onset<=times[-1]:state=Eligibility.POSITIVE
        elif t+6<=times[-1] and (onset is None or onset>t+6):state=Eligibility.NEGATIVE
        else:state=Eligibility.INCOMPLETE
        value=1 if state is Eligibility.POSITIVE else 0 if state is Eligibility.NEGATIVE else None
        endpoints.append(Endpoint(j+1,state,value))
    return Adjudication(tuple(endpoints),onset)
