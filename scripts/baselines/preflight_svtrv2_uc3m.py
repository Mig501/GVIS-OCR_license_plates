#!/usr/bin/env python3
"""Preflight CPU (sin CUDA) de SVTRv2-Tiny/Small sobre UC3M-LP/test."""
from run_svtrv2_uc3m_baselines import main

if __name__ == '__main__':
    raise SystemExit(main(['--check-only']))
