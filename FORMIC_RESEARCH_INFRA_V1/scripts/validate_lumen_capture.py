#!/usr/bin/env python3
"""Run the unchanged V3.1 schema validator with the bundled library on sys.path."""
from pathlib import Path
import sys,runpy
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
runpy.run_path(str(ROOT/'reference/validate_capture_v3_1.py'),run_name='__main__')
