#!/usr/bin/env python3
"""从统一 300 候选池按类别×强度×外观父图公平随机抽取 C1-matched。"""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
import yaml
def stable(x): return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8],"big")
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--mode",choices=("sd21","sdxl"),required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());root=Path(c["output_root"]); rows=[json.loads(x) for x in (root/"candidate_plan.jsonl").read_text().splitlines() if x]; sub="op_i2i" if a.mode=="sd21" else "p1_sdxl_img2img"; by=defaultdict(list)
 for r in rows:
  path=root/sub/r["compound_class"]/f'{r["candidate_id"]}.png'
  if not path.is_file(): raise FileNotFoundError(path)
  r={**r,"output_path":str(path)};by[(r["nutrition_class"],float(r["denoising_strength"]),str(r["appearance_parent_subject_id"]))].append(r)
 selected=[]; quota_records=[]
 for cls in ("malnourished_hand","normal_hand"):
  parents=sorted({k[2] for k in by if k[0]==cls},key=int)
  for si,s in enumerate((.15,.22,.30)):
   target=30; order=parents[si%len(parents):]+parents[:si%len(parents)]; base,extra=divmod(target,len(order)); quotas={x:base+(i<extra) for i,x in enumerate(order)}
   for parent in order:
    candidates=sorted(by[(cls,s,parent)],key=lambda x:x["candidate_id"])
    if len(candidates)<quotas[parent]: raise RuntimeError(f"候选不足:{cls}/{s}/{parent}")
    selected.extend(random.Random(stable(f"{c['classification_seed']}:c1:{cls}:{s}:{parent}")).sample(candidates,quotas[parent])); quota_records.append({"class":cls,"strength":s,"appearance_parent_subject_id":parent,"quota":quotas[parent]})
 if Counter(x["nutrition_class"] for x in selected)!=Counter({"malnourished_hand":90,"normal_hand":90}): raise RuntimeError("抽样数量错误")
 out=root/"selection_c1_matched";out.mkdir(exist_ok=True); (out/"c1_matched_selection.json").write_text(json.dumps({"stage":c["stage"],"seed":c["classification_seed"],"pool_per_class":90,"strata_quotas":quota_records,"conditions":{"c1_raw":selected}},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
