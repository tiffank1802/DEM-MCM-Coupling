"""Extraction compacte des chunks DEM.

Lit les 8 chunks ``data/chunks/simulation_part_*.parquet.gz`` et produit un
fichier NumPy compressé ``data/compact.npz`` contenant, pour chaque ligne :

* ``t``      : indice de pas de temps DEM (int16), recouvré de Fichier_Source ;
* ``pid``    : identifiant de particule (int16) ;
* ``xyz``    : coordonnées (float32, (n, 3)) ;
* ``vnorm``  : norme de la vitesse (float32) ;
* ``small``  : booléen, diamètre 4 mm (True) / 8 mm (False).

Ce format compact (~150 Mo en RAM) permet toutes les études du rapport sans
recharger les parquets.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = sorted((ROOT / "data" / "chunks").glob("simulation_part_*.parquet.gz"))
OUT = ROOT / "data" / "compact.npz"

COLS = [
    "Fichier_Source",
    "Particle_ID",
    "Diameter",
    "coordinates:0",
    "coordinates:1",
    "coordinates:2",
    "Velocity:0",
    "Velocity:1",
    "Velocity:2",
]


def main() -> None:
    ts, pid, xyz, vn, small = [], [], [], [], []
    for i, path in enumerate(CHUNKS):
        print(f"chunk {i + 1}/{len(CHUNKS)} : {path.name}")
        # Chunk = fichier parquet compressé par gzip externe :
        # on décompresse en mémoire avant lecture.
        with gzip.open(path, "rb") as fh:
            df = pd.read_parquet(io.BytesIO(fh.read()), columns=COLS)
        t = (
            df["Fichier_Source"]
            .str.extract(r"(\d+)", expand=False)
            .astype(np.int32)
            .to_numpy()
        )
        ts.append(t.astype(np.int16))
        pid.append(df["Particle_ID"].to_numpy().astype(np.int16))
        xyz.append(
            df[["coordinates:0", "coordinates:1", "coordinates:2"]]
            .to_numpy()
            .astype(np.float32)
        )
        v = df[["Velocity:0", "Velocity:1", "Velocity:2"]].to_numpy()
        vn.append(np.linalg.norm(v, axis=1).astype(np.float32))
        small.append((df["Diameter"].to_numpy() < 0.006))
        del df

    t = np.concatenate(ts)
    order = None  # rows are naturally grouped; keep as-is, sort on load if needed
    np.savez_compressed(
        OUT,
        t=t,
        pid=np.concatenate(pid),
        xyz=np.concatenate(xyz),
        vnorm=np.concatenate(vn),
        small=np.concatenate(small),
    )
    print("écrit :", OUT, f"({OUT.stat().st_size / 1e6:.0f} Mo)")
    print("timesteps:", t.min(), "→", t.max(), "| lignes:", len(t))


if __name__ == "__main__":
    main()
