#!/usr/bin/env python3
"""Permit NVCC13.0 to parse static_asserts inside discarded generic-lambda arms.

The valid template configuration set is unchanged: fold is2 or4, head count
per CTA64 or32. No arithmetic, memory access, or control flow is changed.
"""
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text()
for old,new in [('static_assert(FOLD_FACTOR == 4);','static_assert(FOLD_FACTOR == 2 || FOLD_FACTOR == 4);'),
                ('static_assert(H_Q_PER_CTA == 32);','static_assert(H_Q_PER_CTA == 64 || H_Q_PER_CTA == 32);')]:
    assert src.count(old)==1,old
    src=src.replace(old,new)
Path(sys.argv[2]).write_text(src)
