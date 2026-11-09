#!/usr/bin/env python3
"""Solo comprueba en CPU MDiff4STR-Small/ELPD; nunca ejecuta inferencia."""
import sys
from run_mdiff4str_elpd_baseline import main

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--check-only']
    main()
