"""sys.path bootstrap so flat imports keep working after the folder reorg.

Every runnable script lives in a subfolder (inference/, dataprep/, experiment/,
tables/, figures/, plus the test mains in core/). Shared modules live in the src
root (config), core/ (engine libraries), tables/ and figures/ (topic libraries).

A script puts the src root on sys.path and then does `import _bootstrap`; importing
this module registers the remaining library folders so that flat imports such as
`import config`, `from lane_stitcher import ...`, `import table_common`, or
`import figure_render` resolve regardless of which folder the script lives in.
Same-folder siblings (e.g. infer_common) resolve on their own via the script's
own directory, so inference/ and dataprep/ need no entry here.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))

for _name in ("", "core", "tables", "figures"):
    _path = os.path.join(_SRC, _name) if _name else _SRC
    if _path not in sys.path:
        sys.path.insert(0, _path)
