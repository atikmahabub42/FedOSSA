#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_sweep.py -- Main comparison: test perplexity, CIDEr-D, training loss per round (Table I, Fig. 2).

    python run_sweep.py                                   # paper configuration, all models
    python run_sweep.py --models gpt2-l --methods fedossa  # any subset (see --help)

Runs the models one after another (each in its own process, so GPU memory is freed)
and writes results/part_<model>/sweep_results.csv + sweep_curves.json. Resumable: run the same command again and
finished runs are skipped. All options: python run_sweep.py --child --help
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fedossa_core as fc

if __name__ == "__main__":
    sys.exit(fc.launch("sweep", os.path.abspath(__file__)))
