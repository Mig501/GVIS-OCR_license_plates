#!/usr/bin/env python3
"""Preflight CPU del baseline PARSeq sobre UC3M-LP/test (491 recortes)."""
from run_parseq_uc3m_baseline import main

if __name__ == '__main__':
    raise SystemExit(main(['--check-only']))
