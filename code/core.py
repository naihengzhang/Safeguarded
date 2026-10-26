from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.spatial.distance import cdist

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def distances(q,M):
 q=np.asarray(q);return cdist(q.reshape(-1,q.shape[-1]),M).min(1).reshape(q.shape[:-1])
def scores(q,M,rho=.01):
 d=distances(q,M);k=max(1,int(np.ceil(d.shape[-1]*rho)));return np.partition(d,d.shape[-1]-k,axis=-1)[...,-k:].mean(-1)
def safeguarded_scores(q,B,M,threshold,lam=.5,rho=.01):
 """Residual rank correction with exact preservation of source-threshold decisions."""
 source=scores(q,B,rho);adapted=scores(q,M,rho);corrected=adapted+lam*(source-adapted)
 above=np.nextafter(float(threshold),np.inf)
 return np.where(source>threshold,np.maximum(corrected,above),np.minimum(corrected,threshold))
def coreset(x,k):
 if len(x)<=k:return x.copy()
 ix=int(np.argmax(((x-x.mean(0))**2).sum(1)));ids=[];ds=np.full(len(x),np.inf)
 for _ in range(k):
  ids.append(ix);ds=np.minimum(ds,((x-x[ix])**2).sum(1));ds[ids]=-1;ix=int(np.argmax(ds))
 return x[ids].copy()
@dataclass(frozen=True)
class Config:
 capacity:int=32
 candidate_k:int=2
 witness_capacity:int=8
 min_witness:int=8
 gamma:float=.5
 eta:float=0.
 delta:float=.25
 epsilon:float=.1
 kappa:float=.8
 rho:float=.01
class Gate:
 def __init__(self,B,A,u,v,support,sigma,threshold,mode='full',config=Config(),proposer='high',seed=0,confidence_threshold=None):
  self.B=B.copy();self.B.flags.writeable=False;self.S=np.empty((0,B.shape[1]),dtype=B.dtype)
  self.A=A.copy();self.u=u.copy();self.v=v.copy();self.support=support.copy();self.sigma=float(sigma);assert sigma>=1e-6
  self.threshold=threshold;self.mode=mode;self.cfg=config;self.proposer=proposer;self.rng=np.random.default_rng(seed)
  self.conf=threshold if confidence_threshold is None else confidence_threshold;self.w=[];self.seen=set();self.version=0;self.initial_hash=digest(B)
  raw=self.margin(B);ok=raw>0;self.u=self.u[ok];self.v=self.v[ok];self.support=self.support[ok];self.m0=raw[ok]
 @property
 def memory(self):return np.concatenate([self.B,self.S])
 def margin(self,M):
  if not len(self.u):return np.zeros(0)
  diff=distances(self.v,M)-distances(self.u,M);return (diff*self.support).sum(1)/np.maximum(1,self.support.sum(1))
 def propose(self,x):
  d=distances(x,self.B);k=min(self.cfg.candidate_k,len(x))
  if self.proposer=='high':ix=np.argsort(-d,kind='stable')[:k]
  elif self.proposer=='low':ix=np.argsort(d,kind='stable')[:k]
  else:ix=self.rng.choice(len(x),k,replace=False)
  return np.concatenate([self.S,x[ix]])[-self.cfg.capacity:].copy()
 def step(self,x,identity):
  if identity in self.seen:raise ValueError('Repeated identity')
  self.seen.add(identity);M=self.memory;p=float(scores(x,M));version=self.version;S=self.propose(x);new=np.concatenate([self.B,S]);h=digest(S)
  U=H=L=G=float('nan');eligible=len(self.w)>=self.cfg.min_witness;accept=False
  if eligible and self.mode!='frozen':
   if self.mode=='unconditional':accept=True
   elif self.mode=='confidence':accept=p<=self.conf
   else:
    W=np.stack(self.w);old=scores(W,M);ix=np.argsort(old,kind='stable')[:max(1,int(np.ceil(len(old)*self.cfg.gamma)))];U=float(np.mean(old[ix]-scores(W[ix],new))/self.sigma)
    L=float(np.max(np.maximum(scores(self.A,new)-scores(self.A,M),0))/self.sigma)
    if len(self.m0):
     mm=self.margin(new);H=float(np.max(np.maximum(self.margin(M)-mm,0))/self.sigma);G=float(np.min(mm/self.m0))
    accept=U>=self.cfg.eta and L<=self.cfg.epsilon
    if self.mode in ['local','full']:accept=accept and len(self.m0)>0 and H<=self.cfg.delta
    if self.mode=='full':accept=accept and G>=self.cfg.kappa
  if accept:self.S=S;self.version+=1;assert digest(self.S)==h
  self.w.append(x.copy());self.w=self.w[-self.cfg.witness_capacity:]
  assert digest(self.B)==self.initial_hash and len(self.S)<=self.cfg.capacity
  return dict(id=identity,score_before=p,version_before=version,version_after=self.version,accepted=bool(accept),eligible=eligible,utility=U,local_harm=H,retention_loss=L,min_margin_ratio=G,memory_vectors=len(self.S))
 def feature_bytes(self):return sum(a.nbytes for a in [self.B,self.S,self.A,self.u,self.v])+sum(a.nbytes for a in self.w)
