#!/usr/bin/env python3
"""Transparent Lumen budget facts and conditional capacity scenarios.

Default: arithmetic on the REPORTED inventory, no fabricated sensitivities.
--options PATH: exact additive allocation on caller supplied measured options.
No result of this script is a 27B encoded checkpoint.
"""
from pathlib import Path
import argparse,json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from lumen_ref.budget import align,lowrank_cost,allocate_options,Option

def _module_v2_aux_bytes(q_groups,p_groups,mods):
 act_q=[g for g in q_groups if g['module'] in mods];act_p=[p for p in p_groups if p['module'] in mods]
 active_tensors=sum(g['count'] for g in act_q)+sum(p['count'] for p in act_p)
 scales=sum(g['count']*align(g['rows']*2+g['rows']*math.ceil(g['cols']/512),128) for g in act_q)
 framing=0
 for g in act_q:
  elems=g['rows']*g['cols'];tile_w=262_144 if elems>=50_000_000 else 131_072;nt=math.ceil(elems/tile_w)
  framing+=g['count']*(nt*128+align(nt*4,128)+nt*32)
 headers=128+active_tensors*128+align(14*256,128)
 return scales+framing+headers

def facts():
 inv=json.loads((ROOT/'sources'/'inventaire_rapporte.json').read_text())
 q=inv['quantized_groups'];p=inv['protected_groups']
 def one(vis,mtp,ceiling):
  mods={'text'}|({'vision'} if vis else set())|({'mtp'} if mtp else set())
  quant=sum(g['count']*g['rows']*g['cols'] for g in q if g['module'] in mods)
  protected=sum(g['total_params'] for g in p if g['module'] in mods)
  protected_raw=sum(g['total_params']*g['bytes_per_param'] for g in p if g['module'] in mods)
  prot_count=sum(g['count'] for g in p if g['module'] in mods)
  prot_align_ub=prot_count*127; base_aux=_module_v2_aux_bytes(q,p,mods)
  scenarios=[]
  for aux,reserve,over in [(base_aux,18_500_000,0.0025),(100_000_000,50_000_000,0.02),(150_000_000,100_000_000,0.05)]:
   capacity=ceiling-protected_raw-prot_align_ub-aux-reserve
   scenarios.append({'assumed_aux_bytes':aux,'assumed_guard_bytes':reserve,'assumed_coding_overhead_bpw_NOT_MEASURED':over,'available_symbol_model_bpw':capacity*8/quant-over,'status':'conditional_capacity_NOT_encoded_result_NOT_optimized_quality'})
  return {'ceiling_bytes':ceiling,'unique_params_reported':quant+protected,'quantizable_params_reported':quant,'protected_raw_bytes':protected_raw,'protected_alignment_128B_extra_upper_bound':prot_align_ub,'full_budget_effective_bpw':8*ceiling/(quant+protected),'scenarios':scenarios}
 out={'status':'arithmetic_from_reported_inventory; exact revision headers not re-fetched','Max':one(True,True,10_000_000_000),'Eco':one(True,True,9_000_000_000),'Mini_text':one(False,False,8_000_000_000),'lm_head_rank64_INT8':lowrank_cost(248320,5120,64,bits=8,scale_bytes=(248320+64)*2,metadata_bytes=128),'teacher_top128_bytes_per_position':128*2+128*4+4,'teacher_top128_4194304_positions_bytes':(128*2+128*4+4)*4194304,'teacher_top128_plus_mtp_top32_bytes':((128+32)*6+8)*4194304,'bgcc_all_dense_grams_fp32_bytes':17408*5120*5120*4,'bgcc_one_8x8_float64_gram_bytes':8*8*8}
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--options',type=Path);parser.add_argument('--ceiling',type=int,default=10_000_000_000);parser.add_argument('--output',type=Path,default=ROOT/'results'/'budget_faits_et_scenarios.json');args=parser.parse_args()
 if args.options:
  obj=json.loads(args.options.read_text());groups={g:[Option(**o) for o in v] for g,v in obj['groups'].items()}
  result=allocate_options(groups,args.ceiling,obj.get('fixed_bytes',0))
 else:result=facts()
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
