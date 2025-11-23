Workflow

1. src/preprocess/preprocess.py
    - Preprocesses the unzipped 20news-bydate.tar.gz dataset from http://qwone.com/~jason/20Newsgroups/ into a csv, optional cleaning flag required for classical embedders.
    - Saves data into data/csv/20news-label.csv, 20news-raw.csv, and 20news-clean.csv, must create the directory before running code.

2. src/embed/embed.py
    - Embeds the preprocessed 20news-bydate dataset utilizing specific embedding algorithms into data/embed/embed_{algorithm}.npy and data/embed/ids_{algorithm}.npy
    - These are the embeddings and ids of those embeddings in order

3. src/cluster/cluster.py
    - Executes 5 clustering algorithms. Pass a path to an embedding.npy to perform the clustering. Cluster assignments are stored at {cluster_alg}_{embed_alg}_{hyperparameter}.npy in the same order as the embeddings and its ids
    - Gathers statistics and stores in data/analysis/analysis.csv

