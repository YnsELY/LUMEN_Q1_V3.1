#!/usr/bin/env python3
"""Real CPU end-to-end execution on a synthetic causal fixture only."""
from pathlib import Path
import sys,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from formic_lab.runner import run_experiment,freeze,confirm
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--with-confirmation',action='store_true');a=p.parse_args()
run_experiment(ROOT/'configs/demo_cpu.json',a.out)
if a.with_confirmation:
    freeze(a.out,['A0_local_E8','A1_example_rank8_E8']);confirm(a.out)
print(a.out/'RAPPORT_AUTOMATIQUE.md')
