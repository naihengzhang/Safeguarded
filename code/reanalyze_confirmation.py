from pathlib import Path
import csv,itertools,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
CATS=['candle','chewinggum','fryum','macaroni1','macaroni2','pcb2','pcb3','pcb4','pipe_fryum']

def paired_stats(values):
 values=np.asarray(values,float);rng=np.random.default_rng(20260922)
 boot=np.mean(rng.choice(values,(100000,len(values)),replace=True),axis=1)
 perm=np.array([np.mean(values*np.asarray(signs)) for signs in itertools.product([-1,1],repeat=len(values))])
 return {'mean':float(values.mean()),'ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
         'p_raw':float(np.mean(perm>=values.mean()))}

def holm(items):
 out={};running=0.;m=len(items)
 for i,(name,p) in enumerate(sorted(items,key=lambda item:item[1])):
  running=max(running,(m-i)*p);out[name]=min(1.,running)
 return out

def main():
 primary=pd.read_csv(R/'safeguarded_confirmation.csv');base=pd.read_csv(R/'matched_baseline_confirmation.csv')
 proposal={cat:float(primary[primary.category==cat].proposed_AP.mean()) for cat in CATS}
 cat_delta={cat:proposal[cat]-float(primary[primary.category==cat].frozen_AP.mean()) for cat in CATS}
 frozen_stats=paired_stats(list(cat_delta.values()))
 summary=json.loads((R/'safeguarded_summary.json').read_text())
 summary.update({'category_AP_deltas':cat_delta,'macro_AP_delta':frozen_stats['mean'],
                 'category_bootstrap_95_CI':frozen_stats['ci'],
                 'one_sided_exact_signflip_p':frozen_stats['p_raw'],
                 'signflip_definition':'complete enumeration of all 2^9 sign assignments; no Monte Carlo plus-one correction',
                 'bootstrap_seed':20260922})
 (R/'safeguarded_summary.json').write_text(json.dumps(summary,indent=2))
 comparisons={};methods={}
 for mode in ['unconditional','rate_matched','benefit','local']:
  q=base[base.method==mode]
  methods[mode]={key:float(q[key].mean()) for key in ['AP','AUROC','FPR','TPR','safe_AP','safe_AUROC','safe_FPR','safe_TPR','acceptance']}
  other={cat:float(q[q.category==cat].safe_AP.mean()) for cat in CATS}
  comparisons[mode]=paired_stats([proposal[cat]-other[cat] for cat in CATS])
 for mode,column in [('frozen','frozen_AP'),('direct_full','full_AP'),('confidence_safe','confidence_safe_AP')]:
  other={cat:float(primary[primary.category==cat][column].mean()) for cat in CATS}
  comparisons[mode]=paired_stats([proposal[cat]-other[cat] for cat in CATS])
 adjusted=holm([(name,item['p_raw']) for name,item in comparisons.items()])
 for name in comparisons:comparisons[name]['p_holm']=adjusted[name]
 out={'methods':methods,'proposed':{key:float(primary['proposed_'+key].mean()) for key in ['AP','AUROC','FPR','TPR']},
      'category_paired_comparisons':comparisons,'holm_family':list(comparisons),
      'inference_unit':'category','n_categories':len(CATS),'bootstrap_seed':20260922,
      'signflip_definition':'complete enumeration of all 2^9 sign assignments'}
 (R/'matched_baseline_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
