#!/usr/bin/env python3
"""Entry point for formic_lab.cli candidates; portable from any working directory."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formic_lab.cli import main
if __name__ == '__main__':
    raise SystemExit(main(['candidates'] + sys.argv[1:]))
