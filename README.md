Workflow

## Install necessary requirements
1. Create a local python environment
2. Run `pip3 install -r requirements.txt` or `uv install` depending on your preferred python environment manager

## Creating the files form scratch (unnecessary, most of them are commited)

1. src/preprocess/preprocess.py
    - Preprocesses the unzipped 20news-bydate.tar.gz dataset from http://qwone.com/~jason/20Newsgroups/ into a csv, optional cleaning flag required for classical embedders.
    - Saves data into data/csv/20news-label.csv, 20news-raw.csv, and 20news-clean.csv, must create the directory before running code.

2. src/embed/embed.py
    - Embeds the preprocessed 20news-bydate dataset utilizing specific embedding algorithms into data/embed/embed_{algorithm}.npy and data/embed/ids_{algorithm}.npy
    - These are the embeddings and ids of those embeddings in order

3. src/cluster/cluster.py
    - Executes 5 clustering algorithms. Pass a path to an embedding.npy to perform the clustering. Cluster assignments are stored at {cluster_alg}_{embed_alg}_{hyperparameter}.npy in the same order as the embeddings and its ids
    - Gathers statistics and stores in data/analysis/analysis.csv

> note, please contact bmdowns1 if you need the embedding files for 3large, 3small, or bge-m3 as they are above the git size limit threshold

## Running the visualization algorithms and preparing the data for the server

1. src/vis/vis.py (only for rerunning puropses, the outputed files are commited)
    - Running this file will search through the data/embedding folder to run two separate dimensionality reduction algorithms (tSNE and PaCMAP) on each of the embeddings provided in data/embeddings
2. src/organize/organize.py
    - note: Before running this file make the data/json directory
    - This file will create the JSON files to be served by the server, it takes the vis output, the clustering outputs, and the analysis csv and combines each possible combination together into its own JSON file, named under the schema \<vis>\_\<clustering>\_\<hyperparameter>\_\<embedding> for consistency
    - This will flood the data/json directory with a about 156 of reasonably sized (3mb) JSON files 

## Starting the service

1. Make sure you have docker and docker-compose installed (and the docker daemon running)
    - You can check this by running `docker --version` and `docker-compose --version`
2. `cd` into the root project directory and run `docker-compose up --build`
    - This process will take a while as the image will need to install the necessary python dependencies
3. Head to localhost:3000 in the browser and you should see the service running locally on your machine!


## A couple notes

We appreciate you checking out our project. With several improvements to come, we believe that allowing users to examine the actual findings of clustering algorithms through human-oriented exploratory search provides a novel solution to cluster analysis and assessment. 


