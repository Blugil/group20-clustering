import numpy as np
import os
import matplotlib.pyplot as plt
import pacmap
from pathlib import Path
from sklearn.manifold import TSNE


def plot(embeddings_2d, clusters):
    plt.figure(figsize=(8, 6))

    # If the cluster labels are non-numeric, map them to integer codes for coloring.
    if not np.issubdtype(clusters.dtype, np.number):
        unique_labels, label_indices = np.unique(clusters, return_inverse=True)
        color_values = label_indices
    else:
        color_values = clusters

    # the proverbial rendering of datapoints
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=color_values,
        s=10,
        alpha=0.8,
    )

    plt.title("t-SNE Visualization of Embeddings (Colored by Cluster)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.show()




def tsne(dimension=3):
    #TODO: temporary imports, real vis work will involve creating a tsne or pacmap output for every cluster combo 
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    EMBED_DIR = DATA_DIR / "embed"
    CLUSTER_DIR = DATA_DIR / "cluster"
    VIS_PATH = DATA_DIR / "vis"

    # relevant later
    # ids_path = os.path.join("../../embed/", "ids_minilm.npy")

    if not os.path.isdir(VIS_PATH):
        print("create the vis folder\naborting..")
        exit()

    # creates a tsne reduction for each embedding in the folder
    embed_files = [f.name for f in (EMBED_DIR).iterdir() if f.is_file() and f.name.split("_")[0] == "embed"]
    for file in embed_files:
        # file names are in the structure of:
        # cluster type, embedding type, hyperparameter
        embedding_algo = file.split("_")[1].split(".")[0]

        if Path.exists(VIS_PATH / f"tsne_{embedding_algo}.npy"):
            print(f"tsne_{embedding_algo}.npy already exists, no need to recompute")
            continue

        embeddings = np.load(EMBED_DIR / file)
        tsne = TSNE(
            n_components=dimension,
            perplexity=50.0,   
            learning_rate=200.0,
            init="pca",
            random_state=42,
            verbose=1
        )

        embeddings_2d = tsne.fit_transform(embeddings)

        # Save the 2D t-SNE output to a new .npy file
        np.save(VIS_PATH / f"tsne_{embedding_algo}.npy", embeddings_2d)
        print("Saved t-SNE output to tsne_output.npy")



def pmap(dimension=3):
    #TODO: temporary imports, real vis work will involve creating a tsne or pacmap output for every cluster combo 
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    EMBED_DIR = DATA_DIR / "embed"
    CLUSTER_DIR = DATA_DIR / "cluster"
    VIS_PATH = DATA_DIR / "vis"

    # relevant later
    # ids_path = os.path.join("../../embed/", "ids_minilm.npy")

    if not os.path.isdir(VIS_PATH):
        print("create the vis folder\naborting..")
        exit()

    # creates a tsne reduction for each embedding in the folder
    embed_files = [f.name for f in (EMBED_DIR).iterdir() if f.is_file() and f.name.split("_")[0] == "embed"]
    for file in embed_files:
        # file names are in the structure of:
        # cluster type, embedding type, hyperparameter
        embedding_algo = file.split("_")[1].split(".")[0]

        if Path.exists(VIS_PATH / f"pmap_{embedding_algo}.npy"):
            print(f"pmap_{embedding_algo}.npy already exists, no need to recompute")
            continue

        embeddings = np.load(EMBED_DIR / file)

    # testing pacmap visualization
        pmap = pacmap.PaCMAP(
            n_components=dimension, 
            n_neighbors=10, 
            MN_ratio=0.5, 
            FP_ratio=2.0,
            verbose=True,
            num_iters=450
        ) 

        embeddings_2d = pmap.fit_transform(embeddings)

        # Save the 2D t-SNE output to a new .npy file
        np.save(VIS_PATH / f"pmap_{embedding_algo}.npy", embeddings_2d)
        print(f"Saved PacMAP output to pmap_{embedding_algo}.npy")


if __name__ == "__main__":
    # compute the tsne and pmap reductions
    tsne()
    pmap()