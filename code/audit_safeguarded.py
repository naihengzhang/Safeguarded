import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
d=pd.read_csv(R/'safeguarded_confirmation.csv')
max_error=0.;decision_violations=0;files=0
for row in d.itertuples(index=False):
 tag=f'{row.category}_s{row.seed}_b{row.brightness}_c{row.contamination:g}_safeguarded'
 z=np.load(R/'predictions'/f'{tag}.npz');y=z['labels'];tau=float(z['threshold'])
 for name,col in [('proposed','proposed_scores'),('frozen','frozen_scores'),('full','full_scores'),
                  ('confidence','confidence_scores'),('confidence_safe','confidence_safe_scores')]:
  p=z[col];checks=[average_precision_score(y,p),roc_auc_score(y,p),
                   np.mean(p[y==0]>tau),np.mean(p[y==1]>tau)]
  saved=[getattr(row,f'{name}_AP'),getattr(row,f'{name}_AUROC'),
         getattr(row,f'{name}_FPR'),getattr(row,f'{name}_TPR')]
  max_error=max(max_error,float(np.max(np.abs(np.array(checks)-saved))))
 decision_violations+=int(np.sum((z['proposed_scores']>tau)!=(z['frozen_scores']>tau)))
 files+=1
out={'prediction_files_checked':files,'metric_max_abs_error':max_error,
     'proposed_vs_frozen_decision_violations':decision_violations,
     'passed':max_error<1e-12 and decision_violations==0}
(R/'safeguarded_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
assert out['passed']
