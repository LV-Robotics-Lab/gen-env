"""Isolated, time-bounded CoACD worker for derived collision geometry."""

import json
import sys

import coacd
import numpy as np


def main():
    source, settings, destination = sys.argv[1:]
    data = np.load(source)
    options = json.loads(open(settings).read())
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(data["vertices"], data["faces"]), **options)
    arrays = {}
    for i, (vertices, faces) in enumerate(parts):
        arrays[f"vertices_{i}"] = vertices
        arrays[f"faces_{i}"] = faces
    np.savez_compressed(destination, **arrays)


if __name__ == "__main__":
    main()
