from pathlib import Path
import argparse,tarfile,json,hashlib,time
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import torch
import torch.nn.functional as F
from torchvision.models import resnet18,ResNet18_Weights
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'data_cache';R=ROOT/'results'
CATS=['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
def load(p,b=1.):return np.asarray(ImageEnhance.Brightness(ImageOps.pad(Image.open(p).convert('RGB'),(256,256),method=Image.Resampling.BILINEAR,color=(0,0,0))).enhance(b)).copy()
def parse_args():
 p=argparse.ArgumentParser(description='Extract VisA and build deterministic feature caches.')
 p.add_argument('--archive',type=Path,default=ROOT/'data'/'VisA_20220922.tar',help='Path to the user-downloaded official VisA tar archive.')
 p.add_argument('--force',action='store_true',help='Regenerate feature caches even when deterministic caches already exist.')
 return p.parse_args()
def main():
 args=parse_args();archive=args.archive.expanduser().resolve();C.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 if not archive.is_file():
  raise FileNotFoundError(f'VisA archive not found: {archive}\nDownload VisA_20220922.tar from the official links in data/README.md and pass --archive if stored elsewhere.')
 torch.set_num_threads(4);torch.manual_seed(0)
 with tarfile.open(archive) as tf:
  for m in tf:
   pp=Path(m.name).parts
   if any(c in pp for c in CATS) and m.isfile() and '..' not in pp and not Path(m.name).is_absolute():tf.extract(m,C/'raw',filter='data')
 model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval();proj=np.random.default_rng(2026).normal(size=(384,32)).astype('float32')/np.sqrt(32);np.save(R/'projection.npy',proj)
 manifest=json.loads((R/'data_manifest.json').read_text()) if (R/'data_manifest.json').exists() else {'encoder':'resnet18.IMAGENET1K_V1','input':256,'grid':8,'projection_dim':32,'projection_seed':2026,'torch':torch.__version__,'categories':{}}
 def extract(images):
  ans=[]
  with torch.inference_mode():
   for start in range(0,len(images),16):
    a=np.stack(images[start:start+16]);x=torch.from_numpy(a.transpose(0,3,1,2).copy()).float()/255;x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    h=model.maxpool(model.relu(model.bn1(model.conv1(x))));h=model.layer1(h);a2=model.layer2(h);a3=model.layer3(a2);h=torch.cat([F.adaptive_avg_pool2d(a2,(8,8)),F.adaptive_avg_pool2d(a3,(8,8))],1)
    z=h.permute(0,2,3,1).reshape(len(a),64,384).numpy()@proj;z/=np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8);ans.append(z.astype('float32'))
  return np.concatenate(ans)
 start=time.perf_counter()
 for cat in CATS:
  if not args.force and (C/f'{cat}_features.npz').exists() and cat in manifest['categories']:
   print(cat,'cached',flush=True);continue
  base=next(p for p in (C/'raw').rglob(cat) if p.is_dir());norm=sorted((base/'Data/Images/Normal').glob('*'));ab=sorted((base/'Data/Images/Anomaly').glob('*'));norm=[p for p in norm if p.suffix.lower() in ['.jpg','.png','.jpeg']];ab=[p for p in ab if p.suffix.lower() in ['.jpg','.png','.jpeg']]
  rng=np.random.default_rng(100);normal=[norm[i] for i in rng.permutation(len(norm))[:220]];anomaly=[ab[i] for i in rng.permutation(len(ab))[:100]];paths=normal+anomaly;assert len(paths)==320
  records=[{'index':i,'path':str(p.relative_to(C/'raw')),'label':int(i>=220),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for i,p in enumerate(paths)];assert len({r['sha256'] for r in records})==320;data={}
  for b in [1.,1.2]:data[f'x_{b}']=extract([load(p,b) for p in paths]);print(cat,b,'features done',flush=True)
  for family in ['scratch','patch']:
   images=[]
   for p in normal:
    im=load(p,1.2)
    if family=='scratch':im[108:148,124:132]=255
    else:im[112:144,112:144]=np.indices((32,32)).sum(0)[...,None]%2*255
    images.append(im)
   data[family]=extract(images)
  np.savez_compressed(C/f'{cat}_features.npz',**data);manifest['categories'][cat]=records;(R/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
 manifest['additional_extraction_seconds']=time.perf_counter()-start;manifest['model_sha256']=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest();(R/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
