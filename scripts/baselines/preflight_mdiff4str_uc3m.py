#!/usr/bin/env python3
"""Preflight CPU: MDiff4STR-Small zero-shot sobre UC3M-LP/test (491 PNG)."""
from run_mdiff4str_uc3m_baseline import main

if __name__ == '__main__':
    raise SystemExit(main(['--check-only']))
