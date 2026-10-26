from pathlib import Path
import csv,json
import experiment as e

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'

def main():
 e.MAN=json.loads((R/'data_manifest.json').read_text());lock=json.loads((R/'tuning_lock.json').read_text());rows=[]
 for cat in e.CATS:
  for seed in range(2,7):
   for proposer in ['random','low']:
    rows.append(e.run(cat,seed,1.2,.2,'unconditional',proposer,1.))
    rows.append(e.run(cat,seed,1.2,.2,'rate_matched',proposer,lock['rate_multiplier']))
 with (R/'proposer_completion.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 print('completed proposer runs',len(rows))

if __name__=='__main__':main()
