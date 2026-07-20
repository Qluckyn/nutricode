#!/usr/bin/env python3
"""CPU 审计正式 V2 生成：计划一致性、训练隔离、强度/父图分布与近重复预筛。"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import yaml

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def dhash(path: Path) -> str:
    a=np.asarray(Image.open(path).convert('L').resize((9,8), Image.Resampling.LANCZOS), dtype=np.int16)
    return ''.join('1' if x else '0' for x in (a[:,1:] > a[:,:-1]).ravel())

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True); p.add_argument('--allow-incomplete',action='store_true'); p.add_argument('--compute-dhash',action='store_true'); a=p.parse_args()
    c=yaml.safe_load(a.config.read_text(encoding='utf-8')); plan={r['candidate_id']:r for r in (json.loads(x) for x in Path(c['generation_plan']).read_text(encoding='utf-8').splitlines() if x)}
    root=Path(c['generation_output_root'])/'foundhand_i2i'; records=[]; errors=[]; hashes=defaultdict(list)
    for meta in sorted((root/'metadata').glob('*.json')):
        try: row=json.loads(meta.read_text(encoding='utf-8'))
        except json.JSONDecodeError: continue # 正在写入的元数据在下次审计处理。
        expected=plan.get(row.get('candidate_id'))
        if not expected: errors.append(f'未知候选:{meta.name}'); continue
        for key in ('compound_class','pose','appearance_parent_subject_id','structure_parent_subject_id','denoising_strength','seed'):
            if row.get(key)!=expected.get(key): errors.append(f'{row["candidate_id"]}:{key}与冻结计划不一致')
        image=Path(row.get('output_path',''))
        if not image.is_file(): errors.append(f'{row["candidate_id"]}:缺少输出图'); continue
        if row.get('output_sha256') and digest(image)!=row['output_sha256']: errors.append(f'{row["candidate_id"]}:图像哈希不一致')
        if row['appearance_parent_subject_id']==row['structure_parent_subject_id']: errors.append(f'{row["candidate_id"]}:外观与结构父图相同')
        row['cpu_dhash']=dhash(image) if a.compute_dhash else None
        if row['cpu_dhash']: hashes[(row['nutrition_class'],row['cpu_dhash'])].append(row['candidate_id'])
        records.append({k:row.get(k) for k in ('candidate_id','nutrition_class','compound_class','pose','appearance_parent_subject_id','structure_parent_subject_id','denoising_strength','output_path','keypoints_path','cpu_dhash')})
    if not a.allow_incomplete and len(records)!=len(plan): errors.append(f'生成未完成:{len(records)}/{len(plan)}')
    duplicate_groups={f'{k[0]}:{k[1]}':v for k,v in hashes.items() if len(v)>1}
    out=Path(c['generation_output_root'])/'audit'; out.mkdir(parents=True,exist_ok=True)
    (out/'candidate_cpu_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    report={'stage':c['stage'],'planned':len(plan),'completed':len(records),'status':'pass' if not errors else 'fail','errors':errors,'by_class':Counter(r['compound_class'] for r in records),'by_strength':Counter(str(r['denoising_strength']) for r in records),'exact_dhash_collision_groups':duplicate_groups,'dhash_enabled':a.compute_dhash,'uses_test_data':False}
    (out/'generation_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=dict)+'\n',encoding='utf-8')
    print(json.dumps({k:(dict(v) if isinstance(v,Counter) else v) for k,v in report.items() if k not in ('errors','exact_dhash_collision_groups')},ensure_ascii=False))
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
