
from pathlib import Path
import pandas as pd
import numpy as np
import os
import argparse

from sentence_transformers import SentenceTransformer

def embed_mini_lm():

    # Handle path stuff
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    CSV_DIR = DATA_DIR / "csv"
    EMBED_DIR = DATA_DIR / "embed"
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    input_csv = CSV_DIR / "20news-raw.csv"
    embed_path = EMBED_DIR / "embed_minilm.npy"
    ids_path = EMBED_DIR / "ids_minilm.npy"

    # Read in the CSV 
    df = pd.read_csv(input_csv)

    # Read the corpus and ids as a tolist so we don't shuffle and they are aligned
    corpus = df["content_raw"].astype(str).tolist()
    np_ids = df["id"].astype(str).to_numpy(dtype="U")

    # Perform the embedding
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Starting MiniLM Embedding")

    embedding = model.encode(
        corpus,
        batch_size=64,
        show_progress_bar=True)
    
    # Convert embedding to float32 
    np_embedding = np.asarray(embedding, dtype=np.float32)

    assert embedding.shape[0] == np_ids.shape[0]

    print("\tEmbedding Complete")
    print(f"\t\tSaving embed to {embed_path}")
    print(f"\t\tSaving ids to {ids_path}")

    # Save embeddings and ids in numpy array for fast access
    np.save(ids_path, np_ids)
    np.save(embed_path, np_embedding)

def embed_bge_m3():
    #TODO
    pass

def embed_bert():
    #TODO
    pass

def embed_3small():
    #TODO
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embed", type=str, help="Embedding Mode -> minilm | bge_m3 | 3small | 3large")
    args = parser.parse_args()

    if args.embed == 'minilm':
        embed_mini_lm()
    elif args.embed == 'bge_m3':
        print("Not Implemented")
    elif args.embed == '3small':
        print("Not Implemented")
    elif args.embed == '3large':
        print("Not Implemented")
    else:
        print(f"Unrecognized Mode: {args.embed}")