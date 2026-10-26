from pathlib import Path
import json,csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import experiment as ex
from core import distances
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
COL={'frozen':'#555555','unconditional':'#d55e00','confidence':'#0072b2','rate_matched':'#56b4e9','benefit':'#cc79a7','local':'#e69f00','full':'#009e73'}
LAB={'frozen':'Frozen','unconditional':'Unconditional','confidence':'Tuned confidence','rate_matched':'Rate matched','benefit':'Benefit only','local':'Local probe gate','full':'Full + global floor'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
 fig.savefig(F/f'{name}.pdf',bbox_inches='tight');fig.savefig(F/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(fig)
def static():
 fig,ax=plt.subplots(figsize=(12,7.1));ax.set(xlim=(0,12),ylim=(0,7.1));ax.axis('off')
 edge='#405369';text='#17212b'
 def box(x,y,w,h,txt,c,fs=8.6):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',fc=c,ec=edge,lw=1.25))
  ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=text,linespacing=1.16)
 def arr(a,b,rad=0):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=edge,lw=1.25,connectionstyle=f'arc3,rad={rad}'))
 def route(points):
  xs,ys=zip(*points);ax.plot(xs,ys,color=edge,lw=1.15)
  arr(points[-2],points[-1])
 # Three visually separated stages keep arrows away from text and captions.
 ax.text(.25,6.82,'SOURCE INITIALIZATION',fontsize=9,weight='bold',color='#52677d')
 ax.text(.25,4.70,'PROSPECTIVE ADMISSION',fontsize=9,weight='bold',color='#52677d')
 ax.text(.25,2.50,'DECISION-SAFEGUARDED SCORING',fontsize=9,weight='bold',color='#52677d')
 ax.plot([.25,11.75],[6.67,6.67],color='#cbd5df',lw=.8)
 ax.plot([.25,11.75],[4.55,4.55],color='#cbd5df',lw=.8)
 ax.plot([.25,11.75],[2.35,2.35],color='#cbd5df',lw=.8)

 box(.35,5.35,2.35,.90,'Trusted normal set\nconstruction / anchors / calibration','#edf2f7',8.2)
 box(3.35,5.35,2.25,.90,'Immutable source memory $B$\nfixed threshold $\\tau_{det}$','#dff1ed')
 box(6.35,5.35,2.25,.90,'Adaptive memory $S_t$\nbounded FIFO state','#dff1ed')
 box(9.35,5.35,2.25,.90,'Incoming image $x_t$\npredict before update','#dceef9')
 arr((2.75,5.80),(3.25,5.80));arr((9.25,5.80),(8.70,5.80))

 box(.35,3.18,2.35,.92,'Probe bank\npaired probes + source anchors','#f5edda',8.3)
 box(3.35,3.18,2.25,.92,'Witness buffer $W_t$\ncandidate-disjoint; unlabeled','#edf2f7')
 box(6.35,3.18,2.25,.92,"Shadow memory $S'_t$\ncandidate + FIFO eviction",'#dceef9')
 box(9.35,3.18,2.25,.92,'Prospective gate\nutility + probe harm\nretention + margin floor','#dff1ed',8.2)
 arr((1.52,5.27),(1.52,4.18));arr((7.47,5.27),(7.47,4.18));arr((9.35,3.64),(8.72,3.64))
 route([(1.52,3.10),(1.52,2.91),(9.08,2.91),(9.30,3.42)])
 route([(4.47,3.10),(4.47,2.76),(9.18,2.76),(9.30,3.52)])
 route([(7.47,3.10),(7.47,2.62),(9.28,2.62),(9.38,3.13)])

 box(.35,.82,2.35,.92,'Counterfactual channel\nsource score $s_B(x_t)$','#edf2f7')
 box(3.35,.82,2.35,.92,'Adaptive channel\nmemory score $s_M(x_t)$','#dceef9')
 box(6.60,.82,2.35,.92,'Decision safeguard\nresidual + threshold projection','#f5edda',8.2)
 box(9.65,.82,2.00,.92,'Final anomaly score\nsource decision preserved','#dff1ed',7.8)
 arr((2.80,1.28),(3.23,1.28));arr((5.80,1.28),(6.48,1.28));arr((9.05,1.28),(9.53,1.28))
 route([(4.47,5.27),(3.03,4.48),(3.03,2.18),(1.52,1.84)])
 route([(7.47,5.27),(6.12,4.52),(6.12,2.14),(4.52,1.84)])
 save(fig,'fig1_architecture')
 def unit(t):return np.array([[np.cos(t),np.sin(t)]])
 B=unit(0);u=unit(0);v=unit(1.2);m0=float(distances(v,B)[0]);rows=[];fig,ax=plt.subplots(figsize=(7,4))
 for kind in ['local','full']:
  M=B.copy();vals=[1.]
  for t in range(1,41):
   new=np.concatenate([B,unit(t*.03)]);before=float(distances(v,M)[0]-distances(u,M)[0]);after=float(distances(v,new)[0]-distances(u,new)[0]);yes=before-after<=.031 and (kind=='local' or after>=.8*m0)
   if yes:M=new
   val=float((distances(v,M)[0]-distances(u,M)[0])/m0);vals.append(val);rows.append(dict(rule=kind,step=t,accepted=yes,margin_ratio=val))
  ax.plot(range(41),vals,label='Local only' if kind=='local' else 'Local + global floor',color=COL[kind],lw=2.5)
 ax.axhline(.8,color='#555',ls='--',label='Initial-margin floor');ax.set(xlabel='Proposed replacement',ylabel='Retained / initial probe margin',title='Executed geometric test: probe constraints only');ax.legend();save(fig,'fig2_cumulative')
 with (R/'geometric_stress.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def main():
 static();d=pd.read_csv(R/'metrics.csv');main=d[d.proposer=='high'];shift=main[main.brightness==1.2]
 fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
 for ax,cat in zip(axs,ex.CATS):
  for m in COL:
   q=shift[(shift.category==cat)&(shift.method==m)].groupby('contamination').AP.agg(['mean','std']);ax.plot(q.index*100,q['mean'],marker='o',color=COL[m],label=LAB[m]);ax.fill_between(q.index*100,q['mean']-q['std'],q['mean']+q['std'],alpha=.1,color=COL[m])
  ax.set(title=cat,xlabel='Stream contamination (%)');ax.grid(alpha=.2)
 axs[0].set(ylabel='Image average precision',ylim=(.5,1.));fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3,frameon=False);save(fig,'fig3_AP')
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,b in zip(axs,[1.,1.2]):
  for m in COL:
   q=main[(main.brightness==b)&(main.method==m)].groupby('seed')[['FPR','TPR']].mean();ax.scatter(q.FPR,q.TPR,s=20,color=COL[m],alpha=.4);ax.scatter(q.FPR.mean(),q.TPR.mean(),s=80,color=COL[m],label=LAB[m],edgecolors='white')
  ax.set(title=f'Brightness {b}',xlabel='False-positive rate',ylabel='True-positive rate',xlim=(-.02,1.02),ylim=(-.02,1.02));ax.grid(alpha=.2)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.18),ncol=4,frameon=False);save(fig,'fig4_operating')
 fig,axs=plt.subplots(1,2,figsize=(11,4));q=main.groupby('method').acceptance.mean().reindex(list(COL));bars=axs[0].bar(range(7),q,color=list(COL.values()));axs[0].set_xticks(range(7),[LAB[m] for m in COL],rotation=35,ha='right');axs[0].set_ylabel('Accepted / all stream images');axs[0].set_title('(a) Admission fraction for all methods')
 for bar,val in zip(bars,q):axs[0].text(bar.get_x()+bar.get_width()/2,max(val,0)+.015,f'{val:.2f}',ha='center',va='bottom',fontsize=8)
 for j,m in enumerate(['benefit','local','full']):
  vals=main[main.method==m].min_margin_ratio.dropna();axs[1].scatter(np.full(len(vals),j)+np.linspace(-.12,.12,len(vals)),vals,s=12,alpha=.35,color=COL[m]);axs[1].plot([j-.2,j+.2],[vals.median()]*2,color='black',lw=2)
 axs[1].axhline(.8,color='#555',ls='--');axs[1].set_xticks([0,1,2],[LAB[m] for m in ['benefit','local','full']],rotation=20);axs[1].set_ylabel('Minimum retained / initial margin');axs[1].set_title('(b) Probe-audited methods only; others N/A');fig.tight_layout();save(fig,'fig5_mechanism')
 extra=pd.read_csv(R/'proposer_completion.csv');sub=pd.concat([d[(d.brightness==1.2)&(d.contamination==.2)],extra],ignore_index=True);fig,ax=plt.subplots(figsize=(10,4.6));methods=['unconditional','confidence','rate_matched','benefit','local','full'];width=.13
 for j,m in enumerate(methods):ax.bar(np.arange(3)+(j-2.5)*width,[sub[(sub.method==m)&(sub.proposer==p)].AP.mean() for p in ['high','random','low']],width,color=COL[m],label=LAB[m])
 ax.set_xticks(range(3),['High source distance','Random','Low source distance']);ax.set(ylabel='Mean image AP',ylim=(0,1.02),title='Proposer sensitivity: brightness 1.2; contamination 20%');ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.25),frameon=False);save(fig,'fig6_proposers')
 fig,axs=plt.subplots(1,2,figsize=(11,4));q=main.groupby('method')[['stream_ms_per_image','descriptor_KiB']].mean().reindex(list(COL))
 for ax,key,lab in zip(axs,q.columns,['Gate + pre-update scoring (ms/image)','Persistent descriptor storage (KiB)']):ax.bar(range(7),q[key],color=list(COL.values()));ax.set_xticks(range(7),[LAB[m] for m in COL],rotation=35,ha='right');ax.set_ylabel(lab)
 fig.tight_layout();save(fig,'fig7_resources')
 large=pd.read_csv(R/'resource_control.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,b in zip(axs,[1.,1.2]):
  for j,(m,lab,color) in enumerate([('frozen','Frozen 128 refs',COL['frozen']),('full','Full + audit bank',COL['full']),('large','Frozen 3232 refs',COL['confidence'])]):
   vals=[(large[(large.category==c)&(large.brightness==b)].AP.mean() if m=='large' else main[(main.category==c)&(main.brightness==b)&(main.method==m)].AP.mean()) for c in ex.CATS];ax.bar(np.arange(3)+(j-1)*.23,vals,.23,label=lab,color=color)
  ax.set_xticks(range(3),ex.CATS);ax.set(ylabel='Image AP',ylim=(0,1.02),title=f'Brightness {b}')
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.1),ncol=3,frameon=False);save(fig,'fig8_budget')
 main.groupby('method')[['AP','AUROC','FPR','TPR','acceptance','stream_ms_per_image','descriptor_KiB']].mean().to_csv(R/'summary.csv')
 safe=pd.read_csv(R/'safeguarded_confirmation.csv');ss=json.loads((R/'safeguarded_summary.json').read_text())
 cats=list(ss['category_AP_deltas']);vals=np.array([ss['category_AP_deltas'][c] for c in cats])*100
 fig,ax=plt.subplots(figsize=(10,4.2));colors=np.where(vals>=0,COL['full'],'#c44e52');ax.bar(np.arange(len(cats)),vals,color=colors);ax.axhline(0,color='#333',lw=1);ax.axhline(ss['macro_AP_delta']*100,color='#0072b2',ls='--',label=f"Macro delta = {ss['macro_AP_delta']*100:.3f} pp");ax.set_xticks(np.arange(len(cats)),cats,rotation=30,ha='right');ax.set(ylabel='Proposed - frozen AP (percentage points)',title='Locked confirmation on categories excluded from development');ax.legend(frameon=False);fig.tight_layout();save(fig,'fig10_confirmation_delta')
 matched=json.loads((R/'matched_baseline_summary.json').read_text());cmp=matched['category_paired_comparisons']
 labels=['Full + safeguard','Benefit only','Local probe','Rate matched','Unconditional','Direct full','Frozen','Confidence safe']
 ap=[matched['proposed']['AP'],matched['methods']['benefit']['safe_AP'],matched['methods']['local']['safe_AP'],matched['methods']['rate_matched']['safe_AP'],matched['methods']['unconditional']['safe_AP'],ss['full_AP'],ss['frozen_AP'],ss['confidence_safe_AP']]
 fig,axs=plt.subplots(1,2,figsize=(12,4.7));x=np.arange(len(labels));barcolors=[COL['full'],COL['benefit'],COL['local'],COL['rate_matched'],COL['unconditional'],'#8c6bb1',COL['frozen'],COL['confidence']];delta=(np.asarray(ap)-ss['frozen_AP'])*100
 axs[0].axhline(0,color='#333',lw=1);axs[0].scatter(x,delta,c=barcolors,s=65,zorder=3);axs[0].set_xticks(x,labels,rotation=32,ha='right');axs[0].set_ylabel('AP minus frozen (percentage points)');axs[0].set_title('Locked confirmation: effect size without truncated bars')
 for xi,val in zip(x,delta):axs[0].text(xi,val+.035,f'{val:.3f}',ha='center',va='bottom',fontsize=8)
 order=['frozen','confidence_safe','direct_full','unconditional','rate_matched','benefit','local'];forest_labels=['Frozen','Confidence safe','Direct full','Unconditional','Rate matched','Benefit only','Local probe']
 yy=np.arange(len(order));means=np.array([cmp[m]['mean'] for m in order])*100;lo=np.array([cmp[m]['ci'][0] for m in order])*100;hi=np.array([cmp[m]['ci'][1] for m in order])*100
 axs[1].errorbar(means,yy,xerr=np.vstack([means-lo,hi-means]),fmt='o',color=COL['full'],ecolor='#64748b',capsize=3)
 axs[1].axvline(0,color='#333',lw=1);axs[1].set_yticks(yy,forest_labels);axs[1].invert_yaxis();axs[1].set_xlabel('Full safeguard - comparator AP (percentage points)');axs[1].set_title('Category-paired bootstrap intervals')
 for i,m in enumerate(order):
  if cmp[m]['p_holm']<.05:axs[1].text(hi[i]+.04,i,'Holm p<.05',va='center',fontsize=8,color='#006d5b')
 fig.tight_layout();save(fig,'fig11_confirmation_metrics')
 k=pd.read_csv(R/'ksdd2_confirmation.csv')
 methods=['full_safe','rate_matched_safe','benefit_safe','local_safe','unconditional_safe','confidence_safe','frozen','direct_full']
 labels=['Full gate + safeguard','Rate matched + safeguard','Benefit + safeguard','Local + safeguard','Unconditional + safeguard','Confidence + safeguard','Frozen source','Direct full gate']
 colors=[COL['full'],COL['rate_matched'],COL['benefit'],COL['local'],COL['unconditional'],COL['confidence'],COL['frozen'],'#8c6bb1']
 means=k.groupby('method').AP.mean().reindex(methods)
 seed=k.groupby(['seed','method']).AP.mean().unstack();deltas=(seed['full_safe']-seed['frozen'])*100
 fig,axs=plt.subplots(1,2,figsize=(12,4.7))
 y=np.arange(len(methods));axs[0].barh(y,means,color=colors);axs[0].set_yticks(y,labels);axs[0].invert_yaxis();axs[0].set_xlabel('Image average precision');axs[0].set_xlim(0,1);axs[0].set_title('(a) External KSDD2, 30 locked settings')
 for yi,val in zip(y,means):axs[0].text(val+.008,yi,f'{val:.4f}',va='center',fontsize=8)
 axs[1].axhline(0,color='#333',lw=1);axs[1].scatter(seed.index,deltas,s=70,color=COL['full']);axs[1].plot(seed.index,deltas,color=COL['full'],alpha=.55);axs[1].set_xticks(seed.index);axs[1].set_xlabel('Stream-construction seed');axs[1].set_ylabel('Full safeguard - frozen AP (percentage points)');axs[1].set_title('(b) Seed-blocked effects; shared test identities')
 fig.tight_layout();save(fig,'fig12_ksdd2_external')
if __name__=='__main__':main()
