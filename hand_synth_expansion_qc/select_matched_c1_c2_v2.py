#!/usr/bin/env python3
"""在类别×强度×外观父图的相同配额内，导出 C1 随机池与 C2-QC 优选池。"""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import defaultdict,Counter
from pathlib import Path
import yaml

def stable(text): return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True); p.add_argument('--qc-manifest',type=Path,required=True); a=p.parse_args(); c=yaml.safe_load(a.config.read_text(encoding='utf-8'))
 rows=[json.loads(x) for x in a.qc_manifest.read_text(encoding='utf-8').splitlines() if x]; pool=int(c['classification']['synthetic_pool_per_class']); strengths=[float(x) for x in c['generation']['denoising_strengths']]
 by=defaultdict(list)
 for r in rows: by[(r['nutrition_class'],float(r['denoising_strength']),str(r['appearance_parent_subject_id']))].append(r)
 selected={'c1_raw':[],'c2_qc':[]}; quota_records=[]
 for cls in ('malnourished_hand','normal_hand'):
  parents=sorted({key[2] for key in by if key[0]==cls},key=int)
  if not parents: raise RuntimeError(f'{cls}没有候选')
  for sidx,strength in enumerate(strengths):
   targets=c['classification'].get('synthetic_targets_by_strength', {})
   target=int(targets.get(str(strength), targets.get(strength, pool//len(strengths))))
   order=parents[sidx%len(parents):]+parents[:sidx%len(parents)]
   candidates_by_parent={parent:sorted(by[(cls,strength,parent)],key=lambda r:r['candidate_id']) for parent in order}
   approved_by_parent={parent:[r for r in candidates_by_parent[parent] if r['qc_status']=='approved'] for parent in order}
   if sum(len(v) for v in approved_by_parent.values()) < target:
    raise RuntimeError(f'{cls}/s{strength} QC总候选不足{target}')
   # 先给每个父图相同基准配额；低质较多的父图按实际QC容量截断，余量轮转给仍有容量者。
   base,extra=divmod(target,len(order)); quotas={parent:min(len(approved_by_parent[parent]),base+(rank<extra)) for rank,parent in enumerate(order)}
   remaining=target-sum(quotas.values())
   while remaining:
    progressed=False
    for parent in order:
     if quotas[parent] < len(approved_by_parent[parent]):
      quotas[parent]+=1; remaining-=1; progressed=True
      if not remaining: break
    if not progressed: raise RuntimeError(f'{cls}/s{strength}无法在QC容量内分配目标配额')
   for parent in order:
    quota=quotas[parent]; candidates=candidates_by_parent[parent]; approved=approved_by_parent[parent]
    if len(candidates)<quota: raise RuntimeError(f'{cls}/s{strength}/parent{parent}原始候选不足{quota}')
    rng=random.Random(stable(f'{c["classification_seed"]}:c1:{cls}:{strength}:{parent}')); selected['c1_raw'].extend(rng.sample(candidates,quota)); selected['c2_qc'].extend(sorted(approved,key=lambda r:(-r['qc_score'],r['candidate_id']))[:quota]); quota_records.append({'class':cls,'strength':strength,'appearance_parent_subject_id':parent,'quota':quota,'qc_available':len(approved)})
 for name,items in selected.items():
  if Counter(r['nutrition_class'] for r in items)!={'malnourished_hand':pool,'normal_hand':pool}: raise RuntimeError(f'{name}数量异常')
 if sum(int(c['classification'].get('synthetic_targets_by_strength', {}).get(str(x), c['classification'].get('synthetic_targets_by_strength', {}).get(x, pool//len(strengths)))) for x in strengths) != pool: raise RuntimeError('各强度目标数量之和必须等于每类合成池大小')
 root=Path(c['generation_output_root'])/c.get('selection_dir','selection'); root.mkdir(parents=True,exist_ok=True); payload={'stage':c['stage'],'seed':c['classification_seed'],'pool_per_class':pool,'strata_quotas':quota_records,'conditions':selected}; path=root/'matched_c1_c2_selection.json'; path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'selection':str(path),'counts':{n:dict(Counter(r['nutrition_class'] for r in x)) for n,x in selected.items()}},ensure_ascii=False))
if __name__=='__main__': main()
