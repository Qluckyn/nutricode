#!/usr/bin/env python3
"""为 SD2.1/SDXL 五折无 ControlNet 路线准备 300 候选的 C1-matched 工作区。"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import yaml

METHODS=("sd21_i2i_no_cn","sdxl_i2i_no_cn")
STRENGTHS=(0.15,0.22,0.30)
def rows(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
def seed(base,mode,compound,index): return int.from_bytes(hashlib.sha256(f"{base}:{mode}:{compound}:{index}".encode()).digest()[:8],"big")%(2**31)
def strength(i): return STRENGTHS[i//30] if i<90 else STRENGTHS[(i-90)//70]
def source_root(fold,method):
    if fold==0: return Path("/root/autodl-tmp/runs/hand_synth_expansion_qc/fold_0_pose02/no_controlnet_ablation")/method
    return Path(f"/root/autodl-tmp/runs/hand_synth_expansion_qc/no_controlnet_5fold_seed22/fold_{fold}")/method
def output_subdir(method): return "op_i2i" if method=="sd21_i2i_no_cn" else "p1_sdxl_img2img"
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
 for fold in range(5):
  for method in METHODS:
   old=source_root(fold,method); new=a.output_root/f"fold_{fold}"/method
   base=rows(old/"candidate_plan.jsonl"); grouped={}
   for row in base: grouped.setdefault(row["compound_class"],[]).append(row)
   plan=[]
   for compound,items in sorted(grouped.items()):
    items=sorted(items,key=lambda x:x["candidate_id"]); assert len(items)==90,(fold,method,compound,len(items))
    for i,row in enumerate(items):
     r=dict(row); r["denoising_strength"]=strength(i); r["generation_variant"]=f"strength_{strength(i):.2f}"; plan.append(r)
    for i in range(90,300):
     r=dict(items[i%90]); r["candidate_id"]=f"Q_{r['candidate_id'].split('_')[1]}_{r['candidate_id'].split('_')[2]}_{compound}_{i:04d}"; r["seed"]=seed(22000,r.get("mode","op_i2i"),compound,i); r["denoising_strength"]=strength(i); r["generation_variant"]=f"strength_{strength(i):.2f}"; plan.append(r)
   # SDXL 配置缺少分类字段，继承同 fold 的 SD2.1 分类设定。
   c=yaml.safe_load((old/"config.yaml").read_text()); ref=yaml.safe_load((source_root(fold,"sd21_i2i_no_cn")/"config.yaml").read_text())
   for k in ("fold","data","classification","classification_seed","models","qc","generation","compound_classes","base_seed"): c.setdefault(k,ref[k])
   c.update({"stage":f"C1-matched-fold{fold}-{method}","candidate_count_per_compound":300,"output_root":str(new),"generation_output_root":str(new),"generation_plan":str(new/"candidate_plan.jsonl"),"classifier_data_dir":"classifier_data_c1_matched"})
   new.mkdir(parents=True,exist_ok=True); (new/"candidate_plan.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in plan),encoding="utf-8"); (new/"config.yaml").write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False),encoding="utf-8")
   sub=output_subdir(method)
   for item in plan[:180]:
    # 仅链接原始 90×2 候选，生成器将自动补齐其余图片。
    if int(item["candidate_id"].rsplit("_",1)[-1])>=90: continue
    n=item["candidate_id"]
    for src,dst in ((old/sub/item["compound_class"]/f"{n}.png",new/sub/item["compound_class"]/f"{n}.png"),(old/sub/"metadata"/f"{n}.json",new/sub/"metadata"/f"{n}.json")):
     if not src.is_file(): raise FileNotFoundError(src)
     dst.parent.mkdir(parents=True,exist_ok=True)
     if not dst.exists(): dst.symlink_to(src)
   print(json.dumps({"fold":fold,"method":method,"candidates":len(plan),"output":str(new)},ensure_ascii=False))
if __name__=="__main__": main()
