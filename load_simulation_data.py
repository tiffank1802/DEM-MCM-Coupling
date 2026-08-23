"""
load_simulation_data.py - Auto-généré par split_and_push.py
"""
import pandas as pd
from pathlib import Path


CHUNKS_DIR  = Path("data/chunks")
NUM_CHUNKS  = 15


def load_simulation_data() -> pd.DataFrame:
    """Charge et reconstruit le DataFrame depuis les chunks compressés."""
    chunks = []
    for i in range(NUM_CHUNKS):
        gz_path = CHUNKS_DIR / f"simulation_part_{i:02d}.parquet.gz"
        if not gz_path.exists():
            raise FileNotFoundError(f"Chunk manquant : {gz_path}")
        print(f"📖 Chunk {i+1}/{NUM_CHUNKS} : {gz_path.name}")
        chunks.append(pd.read_parquet(gz_path))

    print("🔗 Assemblage...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


if __name__ == "__main__":
    df = load_simulation_data()
    print(df.head())
    print(df.describe())
