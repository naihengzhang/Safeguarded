from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
COL={'full':'#009e73','benefit':'#cc79a7','local':'#e69f00','rate_matched':'#56b4e9','unconditional':'#d55e00','frozen':'#555555','confidence':'#0072b2'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def main():
 matched=json.loads((R/'matched_baseline_summary.json').read_text());ss=json.loads((R/'safeguarded_summary.json').read_text());cmp=matched['category_paired_comparisons']
 labels=['Full + safeguard','Benefit only','Local probe','Rate matched','Unconditional','Direct full','Frozen','Confidence safe']
 ap=[matched['proposed']['AP'],matched['methods']['benefit']['safe_AP'],matched['methods']['local']['safe_AP'],matched['methods']['rate_matched']['safe_AP'],matched['methods']['unconditional']['safe_AP'],ss['full_AP'],ss['frozen_AP'],ss['confidence_safe_AP']]
 colors=[COL['full'],COL['benefit'],COL['local'],COL['rate_matched'],COL['unconditional'],'#8c6bb1',COL['frozen'],COL['confidence']]
 fig,axs=plt.subplots(1,2,figsize=(12,4.7));x=np.arange(len(labels));delta=(np.asarray(ap)-ss['frozen_AP'])*100
 axs[0].axhline(0,color='#333',lw=1);axs[0].scatter(x,delta,c=colors,s=65,zorder=3);axs[0].set_xticks(x,labels,rotation=32,ha='right');axs[0].set_ylabel('AP minus frozen (percentage points)');axs[0].set_title('Locked confirmation: effect size without truncated bars')
 for xi,val in zip(x,delta):axs[0].text(xi,val+.035,f'{val:.3f}',ha='center',va='bottom',fontsize=8)
 order=['frozen','confidence_safe','direct_full','unconditional','rate_matched','benefit','local'];forest_labels=['Frozen','Confidence safe','Direct full','Unconditional','Rate matched','Benefit only','Local probe']
 yy=np.arange(len(order));means=np.array([cmp[m]['mean'] for m in order])*100;lo=np.array([cmp[m]['ci'][0] for m in order])*100;hi=np.array([cmp[m]['ci'][1] for m in order])*100
 axs[1].errorbar(means,yy,xerr=np.vstack([means-lo,hi-means]),fmt='o',color=COL['full'],ecolor='#64748b',capsize=3)
 axs[1].axvline(0,color='#333',lw=1);axs[1].set_yticks(yy,forest_labels);axs[1].invert_yaxis();axs[1].set_xlabel('Full safeguard - comparator AP (percentage points)');axs[1].set_title('Category-paired bootstrap intervals')
 for i,m in enumerate(order):
  if cmp[m]['p_holm']<.05:axs[1].text(hi[i]+.04,i,'Holm p<.05',va='center',fontsize=8,color='#006d5b')
 fig.tight_layout();fig.savefig(F/'fig11_confirmation_metrics.pdf',bbox_inches='tight');fig.savefig(F/'fig11_confirmation_metrics.png',dpi=170,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
