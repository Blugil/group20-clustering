
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import tiktoken

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


def get_paths():
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    CSV_DIR = DATA_DIR / "csv"
    EMBED_DIR = DATA_DIR / "embed"
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT, CSV_DIR, EMBED_DIR

def load_corpus(CSV_DIR, cleaned=False):
     # Read the corpus and ids as a tolist so we don't shuffle and they are aligned
    if cleaned:
        input_csv = CSV_DIR / "20news-clean.csv"
        df = pd.read_csv(input_csv)
        corpus = df["content_clean"].astype(str).tolist()
        np_ids = df["id"].astype(str).to_numpy(dtype="U")   
    else:
        input_csv = CSV_DIR / "20news-raw.csv"
        df = pd.read_csv(input_csv)
        corpus = df["content_raw"].astype(str).tolist()
        np_ids = df["id"].astype(str).to_numpy(dtype="U")   

    return corpus, np_ids

def chunk_corpus(corpus, encoding, max_tokens):
    chunk_texts = []
    chunk_parents = []
    chunk_token_counts = []

    for doc_idx, text in enumerate(corpus):
        
        tokens = encoding.encode(text)
        n = len(tokens)

        if n == 0:
            chunk_texts.append("")
            chunk_parents.append(doc_idx)
            chunk_token_counts.append(1)
            continue

        start = 0
        while start < n:
            end = min(start + max_tokens, n)
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)

            chunk_texts.append(chunk_text)
            chunk_parents.append(doc_idx)
            chunk_token_counts.append(len(chunk_tokens))

            start = end
    
    chunk_parents = np.array(chunk_parents, dtype=np.int32)
    chunk_token_counts = np.array(chunk_token_counts, dtype=np.int32)

    return chunk_texts, chunk_parents, chunk_token_counts


def embed_lsa():

    # Handle path stuff
    ROOT, CSV_DIR, EMBED_DIR = get_paths()
    corpus, np_ids = load_corpus(CSV_DIR, cleaned=True)
    embed_path = EMBED_DIR / "embed_lsa.npy"
    ids_path = EMBED_DIR / "ids_lsa.npy"

    print("Fitting Vectorizer")
    vectorizer = TfidfVectorizer(
        max_df=.5,
        min_df=3,
        max_features=50000,
        stop_words=None
    )
  
    X_tfidf = vectorizer.fit_transform(corpus)

    print("Performing SVD")
    n_components = 200
    svd = TruncatedSVD(n_components=n_components, random_state=67)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_lsa = lsa.fit_transform(X_tfidf)

    np_embedding = X_lsa.astype(np.float32)
    assert np_embedding.shape[0] == np_ids.shape[0]

    np.save(ids_path, np_ids)
    np.save(embed_path, np_embedding)


def embed_mini_lm():

    # Handle path stuff
    ROOT, CSV_DIR, EMBED_DIR = get_paths()
    corpus, np_ids = load_corpus(CSV_DIR, cleaned=False)
    embed_path = EMBED_DIR / "embed_minilm.npy"
    ids_path = EMBED_DIR / "ids_minilm.npy"
    
    # Perform the embedding
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Starting MiniLM Embedding")
    embedding = model.encode(corpus, batch_size=64, show_progress_bar=True)
    
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

    # Handle path stuff
    ROOT, CSV_DIR, EMBED_DIR = get_paths()
    corpus, np_ids = load_corpus(CSV_DIR, cleaned=False)
    embed_path = EMBED_DIR / "embed_bge-m3.npy"
    ids_path = EMBED_DIR / "ids_bg-m3.npy"
    
    # Perform the embedding
    model = SentenceTransformer("BAAI/bge-m3")
    print("Starting BGE-M3 Embedding")
    embedding = model.encode(corpus, batch_size=8, show_progress_bar=True)
    
    # Convert embedding to float32 
    np_embedding = np.asarray(embedding, dtype=np.float32)
    assert embedding.shape[0] == np_ids.shape[0]
    print("\tEmbedding Complete")
    print(f"\t\tSaving embed to {embed_path}")
    print(f"\t\tSaving ids to {ids_path}")

    # Save embeddings and ids in numpy array for fast access
    np.save(ids_path, np_ids)
    np.save(embed_path, np_embedding)

def embed_3small(limit):

    # Handle path stuff
    ROOT, CSV_DIR, EMBED_DIR = get_paths()
    corpus, np_ids = load_corpus(CSV_DIR, cleaned=False)
    embed_path = EMBED_DIR / "embed_3small.npy"
    ids_path = EMBED_DIR / "ids_3small.npy"

    if limit is not None:
        n = int(len(corpus) * limit)
        corpus = corpus[:n]
        np_ids = np_ids[:n]
        print(f"Using test subset: {n} documents")

    # Set up the open AI client, embedding library, and batching/chunking parameters
    client = OpenAI()
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_CTX_LENGTH = 8191
    EMBEDDING_ENCODING = "cl100k_base"
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    MAX_TOKENS = EMBEDDING_CTX_LENGTH - 128
    BATCH_SIZE = 256

    all_embeddings = []

    chunked_texts, chunk_parents, chunk_token_counts = chunk_corpus(corpus, encoding, MAX_TOKENS)
    num_chunks = len(chunked_texts)
    num_docs = len(corpus)

    pbar = tqdm(total=num_chunks, desc="Embedding chunks", unit="chunk")
    last_status = "N/A"

    for start in range(0, num_chunks, BATCH_SIZE):

        end = min(start + BATCH_SIZE, num_chunks)
        batch = chunked_texts[start:end]

        t0 = time.time()

        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            latency = time.time() - t0
            last_status = f"OK ({latency:.2f}s)"
        except Exception as e:
            latency = time.time() - t0
            last_status = f"ERROR {type(e).__name__})"
            pbar.set_postfix({"last_call": last_status})
            raise

        for item in resp.data:
            all_embeddings.append(item.embedding)

        pbar.update(end - start)
        pbar.set_postfix({"last_call": last_status})

    pbar.close()

    chunk_embeddings = np.asarray(all_embeddings, dtype=np.float32)
    assert chunk_embeddings.shape[0] == num_chunks

    embedding_dim = chunk_embeddings.shape[1]
    doc_sums = np.zeros((num_docs, embedding_dim), dtype=np.float64)
    doc_token_counts = np.zeros(num_docs, dtype=np.int64)

    for i in range(num_chunks):
        doc_idx = int(chunk_parents[i])
        weight = int(chunk_token_counts[i])
        doc_sums[doc_idx] += weight * chunk_embeddings[i]
        doc_token_counts[doc_idx] += weight

    zero_mask = (doc_token_counts == 0)
    if np.any(zero_mask):
        doc_token_counts[zero_mask] = 1

    doc_embeddings = (doc_sums.T / doc_token_counts).T.astype(np.float32)

    assert doc_embeddings.shape[0] == num_docs
    assert doc_embeddings.shape[0] == np_ids.shape[0]

    np.save(ids_path, np_ids)
    np.save(embed_path, doc_embeddings)
    
def embed_3large(limit):

    # Handle path stuff
    ROOT, CSV_DIR, EMBED_DIR = get_paths()
    corpus, np_ids = load_corpus(CSV_DIR, cleaned=False)
    embed_path = EMBED_DIR / "embed_3large.npy"
    ids_path = EMBED_DIR / "ids_3large.npy"

    if limit is not None:
        n = int(len(corpus) * limit)
        corpus = corpus[:n]
        np_ids = np_ids[:n]
        print(f"Using test subset: {n} documents")

    # Set up the open AI client, embedding library, and batching/chunking parameters
    client = OpenAI()
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_CTX_LENGTH = 8191
    EMBEDDING_ENCODING = "cl100k_base"
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    MAX_TOKENS = EMBEDDING_CTX_LENGTH - 128
    BATCH_SIZE = 256

    all_embeddings = []

    chunked_texts, chunk_parents, chunk_token_counts = chunk_corpus(corpus, encoding, MAX_TOKENS)
    num_chunks = len(chunked_texts)
    num_docs = len(corpus)

    pbar = tqdm(total=num_chunks, desc="Embedding chunks", unit="chunk")
    last_status = "N/A"

    for start in range(0, num_chunks, BATCH_SIZE):

        end = min(start + BATCH_SIZE, num_chunks)
        batch = chunked_texts[start:end]

        t0 = time.time()

        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            latency = time.time() - t0
            last_status = f"OK ({latency:.2f}s)"
        except Exception as e:
            latency = time.time() - t0
            last_status = f"ERROR {type(e).__name__})"
            pbar.set_postfix({"last_call": last_status})
            raise

        for item in resp.data:
            all_embeddings.append(item.embedding)

        pbar.update(end - start)
        pbar.set_postfix({"last_call": last_status})

    pbar.close()

    chunk_embeddings = np.asarray(all_embeddings, dtype=np.float32)
    assert chunk_embeddings.shape[0] == num_chunks

    embedding_dim = chunk_embeddings.shape[1]
    doc_sums = np.zeros((num_docs, embedding_dim), dtype=np.float64)
    doc_token_counts = np.zeros(num_docs, dtype=np.int64)

    for i in range(num_chunks):
        doc_idx = int(chunk_parents[i])
        weight = int(chunk_token_counts[i])
        doc_sums[doc_idx] += weight * chunk_embeddings[i]
        doc_token_counts[doc_idx] += weight

    zero_mask = (doc_token_counts == 0)
    if np.any(zero_mask):
        doc_token_counts[zero_mask] = 1

    doc_embeddings = (doc_sums.T / doc_token_counts).T.astype(np.float32)

    assert doc_embeddings.shape[0] == num_docs
    assert doc_embeddings.shape[0] == np_ids.shape[0]

    np.save(ids_path, np_ids)
    np.save(embed_path, doc_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embed", type=str, help="Embedding Mode -> minilm | bge_m3 | 3small | 3large | lsa")
    parser.add_argument("-l", "--limit", type=float, default=None, help="Fraction of docs for test purposes")
    args = parser.parse_args()

    if args.embed == 'minilm':
        embed_mini_lm()
    elif args.embed == 'bge_m3':
        embed_bge_m3()
    elif args.embed == '3small':
        embed_3small(args.limit)
    elif args.embed == '3large':
        embed_3large(args.limit)
    elif args.embed == 'lsa':
        embed_lsa()
    else:
        print(f"Unrecognized Mode: {args.embed}")