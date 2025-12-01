import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np


def create_analysis_json() -> None:

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    ANALYSIS_DIR = DATA_DIR / "analysis"
    JSON_DIR = DATA_DIR / "json"

    analysis_df = pd.read_csv(Path.joinpath(ANALYSIS_DIR, "analysis.csv"))

    
    algorithms = analysis_df["algorithm"]
    print(algorithms.array)

    df_clean = analysis_df.dropna(subset=["algorithm"]).replace({float('nan'): None})

    analysis_dict = {
        record["algorithm"]: record
        for record in df_clean.to_dict(orient="records")
    }

    with open(Path.joinpath(JSON_DIR, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, ensure_ascii=False)



def export_vis_json() -> None:

    # various paths #TODO: trim unused later
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    EMBED_DIR = DATA_DIR / "embed"
    CLUSTER_DIR = DATA_DIR / "cluster"
    VIS_DIR = DATA_DIR / "vis"
    JSON_DIR = DATA_DIR / "json"


    vis_dict = {}
    vis_files = [f.name for f in (VIS_DIR).iterdir() if f.is_file()]

    for file in vis_files:
        embed_type = file.split("_")[1].split(".")[0]
        if embed_type not in vis_dict:
            vis_dict[embed_type] = [file]
        else:
            vis_dict[embed_type].append(file)

    # print(vis_dict)

    cluster_dict = {}
    cluster_files = [f.name for f in (CLUSTER_DIR).iterdir() if f.is_file()]

    for file in cluster_files:
        embed_type = file.split("_")[1]
        if embed_type not in cluster_dict:
            cluster_dict[embed_type] = [file]
        else:
            cluster_dict[embed_type].append(file)

    # print(cluster_dict)

    # triple loop is revolting but here we are
    for embedding in vis_dict:

        ids = np.load(EMBED_DIR / f"ids_{embedding}.npy", allow_pickle=True)

        for vis in vis_dict[embedding]:

            coords = np.load(VIS_DIR / vis) #loads the vis dir
            vis_name = vis.split("_")[0]
            
            for cluster in cluster_dict[embedding]:

                cluster_name = cluster.split("_")[0]
                cluster_hyperparam = cluster.split("_")[2].split(".")[0]

                clusters = np.load(CLUSTER_DIR / cluster, allow_pickle=True)

                # sanity check code
                n_points = coords.shape[0]
                if ids.shape[0] != n_points:
                    raise ValueError(
                        f"ids length ({ids.shape[0]}) does not match coords rows ({n_points})"
                    )

                if clusters is not None and clusters.shape[0] != n_points:
                    raise ValueError(
                        f"clusters length ({clusters.shape[0]}) does not match coords rows ({n_points})"
                    )

                meta = {
                    "emebdding": embedding,
                    "coordinate_dim": int(coords.shape[1]),
                    "num_points": int(n_points),
                    "clustering": cluster_name,
                    "clustering_hyperparam": cluster_hyperparam,
                    "vis_technique": vis_name,
                    "extra": {},  
                }

                points = []

                for i in range(n_points):
                    # Convert id to a JSON-serializable Python type
                    # feels useless 
                    raw_id = ids[i]
                    if isinstance(raw_id, np.generic):
                        point_id = raw_id.item()
                    else:
                        point_id = raw_id  # already a Python scalar/str

                    point = {
                        "id": point_id,
                        "coordinates": coords[i].astype(float).tolist(),
                        "attributes": {},  # per-point extension slot
                    }

                    raw_cluster = clusters[i]
                    if isinstance(raw_cluster, np.generic):
                        point["cluster"] = raw_cluster.item()
                    else:
                        point["cluster"] = raw_cluster

                    points.append(point)
                
                data = {
                    "meta": meta,
                    "points": points,
                }

                with open(JSON_DIR / f"{vis_name}_{cluster_name}_{cluster_hyperparam}_{embedding}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)

                print(f"Wrote visualization JSON to {vis_name}_{cluster_name}_{cluster_hyperparam}_{embedding}.json")


    # WHAT NEEDS TO HAPPEN
    # i need a json output of each cluster that matches the algorithm type on each vis
    # example: tsne_minilm -> each minilm clustering
    # read in the names of all files in the vis folder
    # sort into embedding type + vis tpye
    # [{vis: tsne, embedding: [minilm, etc]}, {vis: pacmap, embedding: [minilm, etc]}]
    # for object in vis_array
    # for each embedding 
    # for each cluster in cluster_folder with matching emebedding
    # do point thing



if __name__ == "__main__":
    create_analysis_json()
    # export_vis_json()