#!/usr/bin/env python3
"""构建正式 V2 的 C0/C1/C2 pose02 分类目录；C1/C2只接受同一分层选择清单。"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import yaml

CLASSES=("malnourished_hand","normal_hand")
def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def link(source:Path,target:Path)->None:
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() or target.is_symlink():
  if not target.is_symlink() or target.resolve()!=source.resolve(): raise FileExistsError(target)
  return
 target.symlink_to(source)
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True); p.add_argument('--condition',choices=('c0','c1_raw','c2_qc'),required=True); p.add_argument('--selection-manifest',type=Path); a=p.parse_args(); c=yaml.safe_load(a.config.read_text(encoding='utf-8')); root=Path(c['output_root']); out=root/c.get('classifier_data_dir','classifier_data_v2')/a.condition/'train'; fold=f"fold_{c['fold']}"
 real=[]
 for cls in CLASSES:
  paths=sorted((Path(c['data']['root'])/fold/'train'/cls).glob('*_02.png'),key=lambda x:int(x.stem.split('_')[0]))
  if len(paths) < int(c['classification']['real_per_class_per_epoch']): raise RuntimeError(f'{cls}真实训练图不足:{len(paths)}')
  for path in paths: link(path,out/cls/path.name); real.append(str(path))
 selected=[]
 if a.condition!='c0':
  if not a.selection_manifest: raise ValueError('C1/C2必须提供--selection-manifest')
  selected=json.loads(a.selection_manifest.read_text(encoding='utf-8'))['conditions'][a.condition]
  pool=int(c['classification']['synthetic_pool_per_class'])
  if Counter(x['nutrition_class'] for x in selected)!=Counter({CLASSES[0]:pool,CLASSES[1]:pool}): raise RuntimeError('选择清单每类数量不符')
  for row in selected:
   src=Path(row['output_path'])
   if not src.is_file(): raise FileNotFoundError(src)
   # 新标记仅供正式V2采样器识别，前缀仍保留真实外观父受试者ID。
   name=f"{row['appearance_parent_subject_id']}_02__pose02_synth__{row['candidate_id']}.png"
   link(src,out/row['nutrition_class']/name)
 report={'stage':c['stage'],'condition':a.condition,'fold':c['fold'],'seed':c['classification_seed'],'pose':'02','real_available':{'malnourished_hand':12,'normal_hand':42},'real_per_class_per_epoch':c['classification']['real_per_class_per_epoch'],'synthetic_per_class_per_epoch':c['classification']['synthetic_per_class_per_epoch'],'synthetic_selected_per_class':dict(Counter(x['nutrition_class'] for x in selected)),'selection_manifest_sha256':sha256_file(a.selection_manifest) if a.selection_manifest else None,'selected_candidates':selected}
 path=out.parent/'pool_manifest.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:v for k,v in report.items() if k!='selected_candidates'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
