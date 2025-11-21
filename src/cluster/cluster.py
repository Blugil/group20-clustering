# Statistics Import
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import entropy

# Clustering Import
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
import hdbscan

# Dimensionality and Representation Import
from sklearn.decomposition import PCA
import numpy as np

# Quality of Life Import
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'",category=FutureWarning)

# User Import
import argparse

# The analysis column header
ANALYSIS_COLUMNS = [
            "algorithm", "nmi_full", "nmi_super", "nmi_sub",
            "ari_full", "ari_super", "ari_sub", "purity_full",
            "purity_super", "purity_sub", "sil", "ch",
            "db", "num_clusters", "noise_frac", "cluster_size_mean",
            "cluster_size_std", "cluster_size_entropy", "pca"
        ]

# Hyperparameters
K_MEANS_ClUSTERS = [6, 12, 20]
HDB_MIN_CLUSTER_SIZE = [10, 20, 40, 60, 100]
DBSCANS_EPS = [0.6, 0.7, 0.8, 0.9]
AGG_N_CLUSTERS = [6, 12, 20] # super, sub, full label levels
SPECTRAL_N_CLUSTERS = [6, 12, 20] # super, sub, full label levels

# Create analysis file if not present or damaged file
def _init_analysis_file(analysis_file_path):

    if not analysis_file_path.exists() or analysis_file_path.stat().st_size == 0:
        df = pd.DataFrame(columns=ANALYSIS_COLUMNS)
        df.to_csv(analysis_file_path, index=False)

# Common paths in project directory we will access for read/write
def _common_paths(embedding_path):
    # Handle Path Stuff
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    CSV_DIR = DATA_DIR / "csv"
    CLUSTER_DIR = DATA_DIR / "cluster"
    ANALYSIS_DIR = DATA_DIR / "analysis"
    labels_csv_path = CSV_DIR / "20news-label.csv"
    embedding_path = Path(embedding_path).resolve()
    prefix = embedding_path.stem.split("_")[1]
    ids_path = embedding_path.with_name(f"ids_{prefix}.npy")
    
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    analysis_file_path = ANALYSIS_DIR / "analysis.csv"
    _init_analysis_file(analysis_file_path)

    return embedding_path, ids_path, labels_csv_path, analysis_file_path, prefix, CLUSTER_DIR

# Calculate the ideal number of components for PCA
def calculate_pca_components(data, variance_threshold=.95):
    pca = PCA().fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_raw = np.searchsorted(cumulative_variance, variance_threshold) + 1
    return n_components_raw

def purity(y_true, y_pred):

    clusters = {}
    correct = 0

    for y_true_0, y_pred_0 in zip(y_true, y_pred):
        clusters.setdefault(y_pred_0, []).append(y_true_0)

    for cluster_labels in clusters.values():
        if not cluster_labels:
            continue

        most_common_count = Counter(cluster_labels).most_common(1)[0][1]
        correct += most_common_count

    return correct / len(y_true)

# Internal Metric Calculations
def internal_metrics(embedding, labels):
    
    labels = np.asarray(labels)

    noise_mask = (labels == -1)
    noise_frac = float(noise_mask.mean())

    if noise_mask.any():
        mask = ~noise_mask
        x_valid = embedding[mask]
        labels_valid = labels[mask]
    else:
        x_valid = embedding
        labels_valid = labels

    unique_clusters = np.unique(labels_valid)


    if len(unique_clusters) < 2 or len(labels_valid) < 10:
        return (
            np.nan,
            np.nan,
            np.nan,
            0,
            noise_frac,
            np.nan,
            np.nan,
            np.nan
        )
    
    sil = silhouette_score(x_valid, labels_valid)
    ch = calinski_harabasz_score(x_valid, labels_valid)
    db = davies_bouldin_score(x_valid, labels_valid)

    unique, counts = np.unique(labels_valid, return_counts=True)
    
    num_clusters = int(len(unique))
    cluster_size_mean = float(counts.mean())
    cluster_size_std = float(counts.std())

    probs = counts / counts.sum()
    cluster_size_entropy = float(entropy(probs))

    return (
        sil,
        ch,
        db,
        num_clusters,
        noise_frac,
        cluster_size_mean,
        cluster_size_std,
        cluster_size_entropy
    )

# General Analysis procedure
def analysis(embedding, labels, labels_df):

    metrics = {}

    y_pred = np.asarray(labels)

    y_full = labels_df["label"].astype(str).values
    y_super = labels_df["superlabel"].astype(str).values
    y_sub = labels_df["sublabel"].astype(str).values

    # Ground Truth Metrics
    metrics["nmi_full"] = normalized_mutual_info_score(y_full, y_pred)
    metrics["nmi_super"] = normalized_mutual_info_score(y_super, y_pred)
    metrics["nmi_sub"] = normalized_mutual_info_score(y_sub, y_pred)

    metrics["ari_full"] = adjusted_rand_score(y_full, y_pred)
    metrics["ari_super"] = adjusted_rand_score(y_super, y_pred)
    metrics["ari_sub"] = adjusted_rand_score(y_sub, y_pred)

    metrics["purity_full"] = purity(y_full, y_pred)
    metrics["purity_super"] = purity(y_super, y_pred)
    metrics["purity_sub"] = purity(y_sub, y_pred)

    sil, ch, db, num_clusters, noise_frac, cluster_size_mean, cluster_size_std, cluster_size_entropy = internal_metrics(embedding, y_pred)

    metrics['sil'] = sil
    metrics['ch'] = ch
    metrics['db'] = db
    metrics['num_clusters'] = num_clusters
    metrics['noise_frac'] = noise_frac
    metrics['cluster_size_mean'] = cluster_size_mean
    metrics['cluster_size_std'] = cluster_size_std
    metrics['cluster_size_entropy'] = cluster_size_entropy

    return metrics

# Write out current metrics to the analysis file
def append_metrics(analysis_file_path, metrics_row):
    normalized_row = {col : metrics_row.get(col, np.nan) for col in ANALYSIS_COLUMNS}
    df_new = pd.DataFrame([normalized_row])
    df_new.to_csv(analysis_file_path, mode="a", header=False, index=False)

# Sanity check we read stuff into the script in the right order
def validate_read(np_embedding, labels_df, np_ids):
    # Sanity check nothing was shuffled and counts are correct
    df_ids = labels_df["id"].astype(str).values
    ids_str = np_ids.astype(str)

    if np_embedding.shape[0] != df_ids.shape[0]:
        raise ValueError("Row Mismatch between embeddings and labels_csv")
    if not np.array_equal(df_ids, ids_str):
        raise ValueError("ID Order mismatch between embeddings and labels CSV")

# 5 different clustering algorithms with statistics collection
def cluster(embedding_path):

    # Determine relevant paths
    embedding_path, ids_path, labels_csv_path, analysis_file_path, prefix, CLUSTER_DIR = _common_paths(embedding_path)

    # Read in the relevant files
    np_embedding = np.load(embedding_path)
    np_ids = np.load(ids_path)
    labels_df = pd.read_csv(labels_csv_path)
    validate_read(np_embedding, labels_df, np_ids)

    # Principal Component Analysis - Reduce the embedding size to a reasonable number
    n_components = calculate_pca_components(np_embedding)
    pca = PCA(n_components=n_components, random_state=67)
    reduced_embedding = pca.fit_transform(np_embedding)

    # Execute KMeans
    for k_value in tqdm(K_MEANS_ClUSTERS, desc=f"K-Means-{prefix}"):
        
        km = KMeans(n_clusters=k_value, n_init=10, random_state=67)
        kmeans_labels = km.fit_predict(reduced_embedding)

        col_name = f"kmeans_{prefix}_k{k_value}"
        out_path = CLUSTER_DIR / f"{col_name}.npy"

        labels_df[col_name] = kmeans_labels
        np.save(out_path, kmeans_labels)

        metrics = analysis(np_embedding, kmeans_labels, labels_df)
        metrics_row = { "algorithm": col_name, "pca": n_components, **metrics}
        append_metrics(analysis_file_path, metrics_row)

    # Execute HDB
    for min_cluster_size in tqdm(HDB_MIN_CLUSTER_SIZE, desc=f"HDBScan-{prefix}"):

        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, metric='euclidean', core_dist_n_jobs=1, cluster_selection_method="eom")
        hdb_labels = hdb.fit_predict(reduced_embedding)

        col_name = f"hdbscan_{prefix}_mcs{min_cluster_size}"
        out_path = CLUSTER_DIR / f"{col_name}.npy"

        labels_df[col_name] = hdb_labels
        np.save(out_path, hdb_labels)

        metrics = analysis(np_embedding, hdb_labels, labels_df)
        metrics_row = { "algorithm": col_name, "pca": n_components, **metrics}
        append_metrics(analysis_file_path, metrics_row)
     
    # Execute DB
    for eps_value in tqdm(DBSCANS_EPS, desc=f"DBSCAN-{prefix}"):
        
        db = DBSCAN(eps=eps_value, min_samples=10, metric="euclidean", n_jobs=1)
        db_labels = db.fit_predict(reduced_embedding)

        eps_str = str(eps_value).replace(".", "p")
        col_name = f"dbscan_{prefix}_eps{eps_str}"
        out_path = CLUSTER_DIR / f"{col_name}.npy"

        labels_df[col_name] = db_labels
        np.save(out_path, db_labels)

        metrics = analysis(np_embedding, db_labels, labels_df)
        metrics_row = { "algorithm": col_name, "pca": n_components, **metrics}
        append_metrics(analysis_file_path, metrics_row)

    # Execute Aggl
    for n_clusters in tqdm(AGG_N_CLUSTERS, desc=f"Agglomerative-{prefix}"):

        aggl = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="euclidean")
        aggl_labels = aggl.fit_predict(reduced_embedding)

        col_name = f"aggl_{prefix}_n{n_clusters}"
        out_path = CLUSTER_DIR / f"{col_name}.npy"
        
        labels_df[col_name] = aggl_labels
        np.save(out_path, aggl_labels)

        metrics = analysis(np_embedding, aggl_labels, labels_df)
        metrics_row = { "algorithm": col_name, "pca": n_components, **metrics}
        append_metrics(analysis_file_path, metrics_row)

    # Execute Spec
    for n_clusters in tqdm(SPECTRAL_N_CLUSTERS, desc=f"Spectral-{prefix}"):
        
        spec = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=10, assign_labels="kmeans", n_init=10, random_state=67)
        spec_labels = spec.fit_predict(reduced_embedding)

        col_name = f"spec_{prefix}_n{n_clusters}"
        out_path = CLUSTER_DIR / f"{col_name}.npy"
        
        labels_df[col_name] = spec_labels
        np.save(out_path, spec_labels)

        metrics = analysis(np_embedding, spec_labels, labels_df)
        metrics_row = { "algorithm": col_name, "pca": n_components, **metrics}
        append_metrics(analysis_file_path, metrics_row)
    
    # Save the labels to csv for redundancy/readability
    labels_df.to_csv(labels_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embed", type=str, help="Path to embedding.npy file")
    args = parser.parse_args()
    cluster(args.embed)
