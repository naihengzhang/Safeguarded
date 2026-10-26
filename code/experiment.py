from pathlib import Path
import json,csv,time,platform,sys,importlib.metadata
from dataclasses import asdict
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from core import Gate,Config,coreset,distances,scores
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';C=ROOT/'data_cache';CATS=['cashew','capsules','pcb1'];SHIFTS=[1.,1.2];RATES=[0,.1,.2];CFG=Config();cache={}
for n in ['predictions','traces','splits']:(R/n).mkdir(exist_ok=True)
MAN=None

def setting(cat,seed):
 key=(cat,seed)
 if key in cache:return cache[key]
 f=np.load(C/f'{cat}_features.npz');rng=np.random.default_rng(seed);ni=rng.permutation(220);ai=rng.permutation(np.arange(220,320));s={'source':ni[:60].tolist(),'anchors':ni[60:68].tolist(),'calibration':ni[68:92].tolist(),'adapt_normal':ni[92:156].tolist(),'eval_normal':ni[156:].tolist(),'adapt_anomaly':ai[:20].tolist(),'eval_anomaly':ai[20:].tolist()};assert len(set(sum(s.values(),[])))==320
 (R/'splits'/f'{cat}_{seed}.json').write_text(json.dumps(s,indent=2));B=coreset(f['x_1.0'][s['source']].reshape(-1,32),128);A=f['x_1.0'][s['anchors']];cal=f['x_1.0'][s['calibration']];sigma=float(max(np.median(distances(cal,B)),1e-6));tau=float(np.quantile(scores(cal,B),.95,method='higher'))
 u=np.concatenate([f['x_1.2'][s['anchors']]]*2);v=np.concatenate([f['scratch'][s['anchors']],f['patch'][s['anchors']]]);mask=np.zeros((16,8,8),bool);mask[:,2:6,2:6]=True
 cache[key]=(f,s,B,A,u,v,mask.reshape(16,64),sigma,tau);return cache[key]
def run(cat,seed,b,rate,mode,prop='high',mult=1.,save=False,return_gate=False):
 f,s,B,A,u,v,mask,sigma,tau=setting(cat,seed);n=40-round(40*rate);ids=np.array(s['adapt_normal'][:n]+s['adapt_anomaly'][:40-n]);ids=ids[np.random.default_rng(seed+900).permutation(40)];ei=np.array(s['eval_normal']+s['eval_anomaly']);y=(ei>=220).astype(int);X=f[f'x_{b}'];g=Gate(B,A,u,v,mask,sigma,tau,'confidence' if mode in ['confidence','rate_matched'] else mode,CFG,prop,seed,tau*mult);tr=[];start=time.perf_counter()
 for i in ids:
  row=g.step(X[i],MAN['categories'][cat][int(i)]['sha256']);row['step']=len(tr)+1;tr.append(row)
 elapsed=time.perf_counter()-start;t=time.perf_counter();p=scores(X[ei],g.memory);scoretime=time.perf_counter()-t;frozen=scores(X[ei],B);assert np.all(p<=frozen+1e-10)
 ratio=float(np.min(g.margin(g.memory)/g.m0)) if len(g.m0) else float('nan')
 if mode=='full':assert ratio>=CFG.kappa-1e-8
 tag=f'{cat}_s{seed}_b{b}_c{rate}_{mode}_{prop}'
 out={'id':tag,'category':cat,'seed':seed,'brightness':b,'contamination':rate,'method':mode,'proposer':prop,'confidence_multiplier':mult,'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'FPR':float(np.mean(p[y==0]>tau)),'TPR':float(np.mean(p[y==1]>tau)),'acceptance':float(np.mean([r['accepted'] for r in tr])),'min_margin_ratio':ratio,'valid_probes':len(g.m0),'stream_ms_per_image':elapsed*25,'score_ms_per_image':scoretime*1000/len(ei),'descriptor_KiB':g.feature_bytes()/1024,'threshold':tau,'sigma':sigma}
 if save:
  np.savez_compressed(R/'predictions'/f'{tag}.npz',evaluation_ids=ei,labels=y,scores=p,frozen_scores=frozen,stream_ids=ids,stream_labels=(ids>=220).astype(int),stream_scores=np.array([r['score_before'] for r in tr]),threshold=tau)
  with (R/'traces'/f'{tag}.jsonl').open('w') as o:
   for row in tr:o.write(json.dumps({k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()})+'\n')
 return (out,g) if return_gate else out

def write(name,rows):
 with (R/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def main():
 global MAN;MAN=json.loads((R/'data_manifest.json').read_text());config={'config':asdict(CFG),'development_seeds':[0,1],'report_seeds':[2,3,4,5,6],'categories':CATS,'rates':RATES,'brightness':SHIFTS,'stream_length':40,'confidence_grid':[.5,1,1.5,2,3,4],'rate_grid':np.linspace(.75,1.,11).tolist(),'scope':'Exploratory; repeated custom partitions of a shared bounded image universe'};(R/'pilot_config.json').write_text(json.dumps(config,indent=2));dev=[]
 for cat in CATS:
  for seed in [0,1]:
   for b in SHIFTS:
    for rate in RATES:
     dev.append(run(cat,seed,b,rate,'full'))
     for mult in config['confidence_grid']:dev.append(run(cat,seed,b,rate,'confidence',mult=mult))
  print('development',cat,flush=True)
 target=float(np.mean([r['acceptance'] for r in dev if r['method']=='full']));best=max(config['confidence_grid'],key=lambda m:np.mean([r['AP'] for r in dev if r['method']=='confidence' and r['confidence_multiplier']==m]));refine=[];stats=[]
 for mult in config['rate_grid']:
  rows=[run(c,s,b,r,'confidence',mult=mult) for c in CATS for s in [0,1] for b in SHIFTS for r in RATES];refine.extend(rows);stats.append({'mult':mult,'acceptance':float(np.mean([r['acceptance'] for r in rows]))})
 matched=min(stats,key=lambda r:(abs(r['acceptance']-target),r['mult']))['mult'];lock={'confidence_multiplier':best,'rate_multiplier':matched,'full_development_acceptance':target,'rate_grid_results':stats,'matching_uses_AP_or_report_labels':False};(R/'tuning_lock.json').write_text(json.dumps(lock,indent=2));write('development_metrics.csv',dev);write('rate_development.csv',refine);print('LOCK',lock,flush=True)
 rows=[]
 for cat in CATS:
  for seed in range(2,7):
   for b in SHIFTS:
    for rate in RATES:
     for mode in ['frozen','unconditional','confidence','rate_matched','benefit','local','full']:rows.append(run(cat,seed,b,rate,mode,mult=best if mode=='confidence' else matched if mode=='rate_matched' else 1.,save=True))
   for prop in ['random','low']:
    for mode in ['confidence','benefit','local','full']:rows.append(run(cat,seed,1.2,.2,mode,prop,best if mode=='confidence' else 1.,save=True))
   write('metrics.csv',rows);print('report',cat,seed,len(rows),flush=True)
 large=[]
 for cat in CATS:
  for seed in range(2,7):
   f,s,B,A,u,v,mask,sigma,tau=setting(cat,seed);src=f['x_1.0'][s['source']].reshape(-1,32);big=src[np.random.default_rng(seed+1100).choice(len(src),3232,replace=False)];th=float(np.quantile(scores(f['x_1.0'][s['calibration']],big),.95,method='higher'));ei=np.array(s['eval_normal']+s['eval_anomaly']);y=(ei>=220).astype(int)
   for b in SHIFTS:
    t=time.perf_counter();p=scores(f[f'x_{b}'][ei],big);dt=time.perf_counter()-t;large.append({'category':cat,'seed':seed,'brightness':b,'AP':average_precision_score(y,p),'FPR':np.mean(p[y==0]>th),'TPR':np.mean(p[y==1]>th),'descriptor_KiB':big.nbytes/1024,'score_ms_per_image':dt*1000/len(ei)})
    np.savez_compressed(R/'predictions'/f'{cat}_{seed}_{b}_large.npz',evaluation_ids=ei,labels=y,scores=p,threshold=th)
 write('resource_control.csv',large)
 versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','pandas','matplotlib','pillow','torch','torchvision']};(R/'environment.json').write_text(json.dumps(dict(packages=versions,python=sys.version,platform=platform.platform()),indent=2));(ROOT/'requirements-lock.txt').write_text('\n'.join(k+'=='+v for k,v in versions.items())+'\n');print('COMPLETE 750 report + 30 resource runs',flush=True)
if __name__=='__main__':main()
