import csv,json,itertools
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
import experiment as e
from core import scores,safeguarded_scores

DEV_CATS=['cashew','capsules','pcb1']
CONFIRM_CATS=['candle','chewinggum','fryum','macaroni1','macaroni2','pcb2','pcb3','pcb4','pipe_fryum']
LAMBDAS=[0,.25,.5,.75,1.,1.5,2.]
REPORT_SEEDS=range(2,7)

def evaluation(cat,seed):
 f,s,B,A,u,v,mask,sigma,tau=e.setting(cat,seed)
 ids=np.array(s['eval_normal']+s['eval_anomaly'])
 return f,s,B,tau,ids,(ids>=220).astype(int)

def measures(y,p,tau,prefix):
 return {prefix+'AP':float(average_precision_score(y,p)),
         prefix+'AUROC':float(roc_auc_score(y,p)),
         prefix+'FPR':float(np.mean(p[y==0]>tau)),
         prefix+'TPR':float(np.mean(p[y==1]>tau))}

def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
 e.MAN=json.loads((e.R/'data_manifest.json').read_text())
 dev=[]
 for cat in DEV_CATS:
  for seed in [0,1]:
   for b in e.SHIFTS:
    for rate in e.RATES:
     _,g=e.run(cat,seed,b,rate,'full',return_gate=True)
     f,s,B,tau,ids,y=evaluation(cat,seed);X=f[f'x_{b}'][ids]
     for lam in LAMBDAS:
      p=safeguarded_scores(X,B,g.memory,tau,lam)
      dev.append({'category':cat,'seed':seed,'brightness':b,'contamination':rate,
                  'lambda':lam,**measures(y,p,tau,'')})
 means={lam:float(np.mean([r['AP'] for r in dev if r['lambda']==lam])) for lam in LAMBDAS}
 lam=max(LAMBDAS,key=lambda x:(means[x],-x))
 lock={'development_categories':DEV_CATS,'development_seeds':[0,1],
       'lambda_grid':LAMBDAS,'mean_AP_by_lambda':means,'selected_lambda':lam,
       'selection_rule':'maximum mean development AP; source-threshold decisions preserved exactly',
       'confirmation_categories':CONFIRM_CATS,'report_seeds':list(REPORT_SEEDS)}
 (e.R/'safeguarded_lock.json').write_text(json.dumps(lock,indent=2))
 write(e.R/'safeguarded_development.csv',dev)

 rows=[];conf_mult=json.loads((e.R/'tuning_lock.json').read_text())['confidence_multiplier']
 for cat in CONFIRM_CATS:
  for seed in REPORT_SEEDS:
   for b in e.SHIFTS:
    for rate in e.RATES:
     full,g=e.run(cat,seed,b,rate,'full',return_gate=True)
     conf,cg=e.run(cat,seed,b,rate,'confidence',mult=conf_mult,return_gate=True)
     f,s,B,tau,ids,y=evaluation(cat,seed);X=f[f'x_{b}'][ids]
     pb=scores(X,B);pm=scores(X,g.memory);pc=scores(X,cg.memory)
     pp=safeguarded_scores(X,B,g.memory,tau,lam);pcs=safeguarded_scores(X,B,cg.memory,tau,lam)
     assert np.array_equal(pp>tau,pb>tau)
     row={'category':cat,'seed':seed,'brightness':b,'contamination':rate,
          'lambda':lam,'full_acceptance':full['acceptance'],'confidence_acceptance':conf['acceptance']}
     for name,p in [('proposed_',pp),('frozen_',pb),('full_',pm),('confidence_',pc),('confidence_safe_',pcs)]:
      row.update(measures(y,p,tau,name))
     rows.append(row)
     tag=f'{cat}_s{seed}_b{b}_c{rate}_safeguarded'
     np.savez_compressed(e.R/'predictions'/f'{tag}.npz',evaluation_ids=ids,labels=y,
                         proposed_scores=pp,frozen_scores=pb,full_scores=pm,
                         confidence_scores=pc,confidence_safe_scores=pcs,threshold=tau)
 write(e.R/'safeguarded_confirmation.csv',rows)

 cat_delta={cat:float(np.mean([r['proposed_AP']-r['frozen_AP'] for r in rows if r['category']==cat])) for cat in CONFIRM_CATS}
 vals=np.array(list(cat_delta.values()));rng=np.random.default_rng(20260921)
 boot=np.mean(rng.choice(vals,(100000,len(vals)),replace=True),axis=1)
 perm=np.array([np.mean(vals*np.array(s)) for s in itertools.product([-1,1],repeat=len(vals))])
 summary={'n_categories':len(vals),'n_paired_settings':len(rows),'category_AP_deltas':cat_delta,
          'macro_AP_delta':float(vals.mean()),'category_bootstrap_95_CI':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
          'one_sided_exact_signflip_p':float(np.mean(perm>=vals.mean()))}
 for method in ['proposed','frozen','full','confidence','confidence_safe']:
  for metric in ['AP','AUROC','FPR','TPR']:
   summary[f'{method}_{metric}']=float(np.mean([r[f'{method}_{metric}'] for r in rows]))
 (e.R/'safeguarded_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({'lock':lock,'summary':summary},indent=2))

if __name__=='__main__':main()
