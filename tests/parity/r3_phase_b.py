# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R3: the R2 harness on GPUs with the PyTorch backend (same comparisons, gates and report).

    python -m tests.parity.r3_phase_b run --fixtures F1 F2 F3 F4 F5 --output OUT --report OUT/r3_report.json -- \\
        --backend torch --torch-device cuda --phase-a-procs 8 --fasttext-model ... --pooled-model ... \\
        --terminal-model ... --extractor ...

No GPU run reproduces production's TPU numerics, so every WARC lands in the report's
``other_device_type`` aggregate; the gates are the same as for TPUs of another type.
"""

from tests.parity.r2_phase_b import main

if __name__ == "__main__":
    main()
