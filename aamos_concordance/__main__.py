"""``python -m aamos_concordance`` : raw-data manifest management.

    python -m aamos_concordance --write-manifest   # hash data/raw and write MANIFEST.sha256
    python -m aamos_concordance --verify           # check the raw files against it
"""

from .data import _main

raise SystemExit(_main())
