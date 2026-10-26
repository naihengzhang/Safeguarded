import csv,json,itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
import experiment as e
from core import scores,safeguarded_scores

R=e.R;CATS=['candle','chewinggum','fryum','macaroni1','macaroni2','pcb2','pcb3','pcb4','pipe_fryum']
MODES=['unconditional','rate_matched','benefit','local']
def met(y,p,t):return [float(average_precision_score(y,p)),float(roc_auc_score(y,p)),float(np.mean(p[y==0]>t)),float(np.mean(p[y==1]>t))]
def stats(v):
 v=np.asarray(v);rng=np.random.default_rng(20260922);boot=np.mean(rng.choice(v,(100000,len(v)),replace=True),1)
 perm=np.array([np.mean(v*np.array(s)) for s in itertools.product([-1,1],repeat=len(v))])
 return {'mean':float(v.mean()),'ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
         'p_raw':float(np.mean(perm>=v.mean()))}
def holm(items):
 out={};run=0.;m=len(items)
 for i,(k,p) in enumerate(sorted(items,key=lambda x:x[1])):run=max(run,(m-i)*p);out[k]=min(1.,run)
 return out
def main():
 e.MAN=json.loads((R/'data_manifest.json').read_text());locks=json.loads((R/'tuning_lock.json').read_text());lam=json.loads((R/'safeguarded_lock.json').read_text())['selected_lambda'];rows=[]
 for cat in CATS:
  for seed in range(2,7):
   for b in e.SHIFTS:
    for rate in e.RATES:
     f,s,B,A,u,v,mask,sigma,tau=e.setting(cat,seed);ids=np.array(s['eval_normal']+s['eval_anomaly']);y=(ids>=220).astype(int);X=f[f'x_{b}'][ids];pb=scores(X,B)
     for mode in MODES:
      mult=locks['rate_multiplier'] if mode=='rate_matched' else 1.
      out,g=e.run(cat,seed,b,rate,mode,mult=mult,return_gate=True);standard=scores(X,g.memory);safe=safeguarded_scores(X,B,g.memory,tau,lam);assert np.array_equal(safe>tau,pb>tau)
      a=met(y,standard,tau);z=met(y,safe,tau);rows.append(dict(category=cat,seed=seed,brightness=b,contamination=rate,method=mode,acceptance=out['acceptance'],AP=a[0],AUROC=a[1],FPR=a[2],TPR=a[3],safe_AP=z[0],safe_AUROC=z[1],safe_FPR=z[2],safe_TPR=z[3]))
 with (R/'matched_baseline_confirmation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 primary=pd.read_csv(R/'safeguarded_confirmation.csv');proposal={c:float(primary[primary.category==c].proposed_AP.mean()) for c in CATS};summary={};comp={}
 for mode in MODES:
  q=[r for r in rows if r['method']==mode];summary[mode]={k:float(np.mean([r[k] for r in q])) for k in ['AP','AUROC','FPR','TPR','safe_AP','safe_AUROC','safe_FPR','safe_TPR','acceptance']};base={c:float(np.mean([r['safe_AP'] for r in q if r['category']==c])) for c in CATS};comp[mode]=stats([proposal[c]-base[c] for c in CATS])
 for mode,col in [('frozen','frozen_AP'),('direct_full','full_AP'),('confidence_safe','confidence_safe_AP')]:
  base={c:float(primary[primary.category==c][col].mean()) for c in CATS};comp[mode]=stats([proposal[c]-base[c] for c in CATS])
 adj=holm([(k,v['p_raw']) for k,v in comp.items()])
 for k in comp:comp[k]['p_holm']=adj[k]
 out={'methods':summary,'proposed':{'AP':float(primary.proposed_AP.mean()),'AUROC':float(primary.proposed_AUROC.mean()),'FPR':float(primary.proposed_FPR.mean()),'TPR':float(primary.proposed_TPR.mean())},'category_paired_comparisons':comp,'holm_family':list(comp)}
 (R/'matched_baseline_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
