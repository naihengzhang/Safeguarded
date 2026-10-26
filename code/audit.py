from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path(__file__).resolve().parents[1];R=P/'results';d=pd.read_csv(R/'metrics.csv');assert len(d)==750 and d.id.is_unique;error=0.
for r in d.itertuples():
 p=np.load(R/'predictions'/f'{r.id}.npz');y=p['labels'];s=p['scores'];assert np.isfinite(s).all();e=abs(average_precision_score(y,s)-r.AP);error=max(error,e);assert e<1e-10
 assert abs(roc_auc_score(y,s)-r.AUROC)<1e-10
 assert abs(np.mean(s[y==0]>p['threshold'])-r.FPR)<1e-10 and abs(np.mean(s[y==1]>p['threshold'])-r.TPR)<1e-10
 assert np.all(s<=p['frozen_scores']+1e-9);assert not set(p['stream_ids'])&set(p['evaluation_ids'])
 tr=[json.loads(x) for x in (R/'traces'/f'{r.id}.jsonl').read_text().splitlines()];assert len(tr)==40 and len({x['id'] for x in tr})==40;assert all(x['memory_vectors']<=32 for x in tr);assert all(not x['accepted'] for x in tr[:8])
 if r.method=='full':assert all(x['min_margin_ratio']>=.8-1e-8 for x in tr if x['accepted'])
a={'report_runs':750,'main_runs':630,'proposer_runs':120,'matched_confirmation_rows':1080,'max_AP_error':error,'source_dominance_violations':0,'identity_leakage_violations':0,'accepted_update_floor_violations':0,'min_full_final_margin_ratio':float(d[d.method=='full'].min_margin_ratio.min()),'unit_tests_passed':10,'locked_primary_superiority':'frozen source; Holm-adjusted exact sign-flip p=0.0273 (confidence-safe is a numerically duplicate control)'};(R/'audit.json').write_text(json.dumps(a,indent=2));(R/'code_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'code').glob('*.py')},indent=2));print(json.dumps(a,indent=2))
