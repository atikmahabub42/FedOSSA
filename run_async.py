#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_async.py -- Robustness to stragglers: test perplexity versus phi (Fig. 3b).

    python run_async.py                                   # paper configuration, all models
    python run_async.py --models gpt2-l --methods fedossa  # any subset (see --help)

Runs the models one after another (each in its own process, so GPU memory is freed)
and writes results/part_<model>/async.csv. Resumable: run the same command again and
finished runs are skipped. All options: python run_async.py --child --help
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fedossa_core as fc

if __name__ == "__main__":
    sys.exit(fc.launch("async", os.path.abspath(__file__)))
