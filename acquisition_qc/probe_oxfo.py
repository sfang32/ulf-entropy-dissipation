"""probe_oxfo.py -- why does OXFO 2015 yield zero usable days?

OXFO is one of the 25 CARISMA sites.  Its 2015 file naming and headers look
normal and 06/12 month first lines print as ordinary spaced records, yet
load_carisma() returned 0 days.  This prints the per-file verdict for a few
representative days so the cause is visible rather than inferred.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "core"))
from run_multiband import _read_carisma_day          # noqa: E402
import numpy as np                                # noqa: E402

D = "/data/ftp/fsf/databases/data_yuliaoku/CARISMA/FGM/1Hz/oxfo/2015"

for lab, p in [("07 filled", D + "/07/20150701OXFO.F01.gz"),
               ("08 glued ", D + "/08/20150801OXFO.F01.gz"),
               ("12 glued ", D + "/12/20151201OXFO.F01.gz")]:
    d = _read_carisma_day(p)
    if d is None:
        print("%-10s -> None (whole day rejected)" % lab)
    else:
        H = np.hypot(d.X.to_numpy(), d.Y.to_numpy())
        print("%-10s -> %6d rows  X=%9.3f Y=%9.3f Z=%9.3f  H=%8.1f  |Z|ok=%s"
              % (lab, len(d), d.X.iloc[0], d.Y.iloc[0], d.Z.iloc[0], H[0],
                 40000.0 < abs(d.Z.iloc[0]) < 90000.0))
