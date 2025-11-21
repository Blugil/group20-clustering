import os
import pandas as pd
from pathlib import Path
import argparse

def split_label_hierarchy(label):
    parts = label.split(".")
    superlabel = parts[0] if parts else label
    sublabel = ".".join(parts[:2]) if len(parts) >= 2 else label
    return superlabel, sublabel


def folder_to_csv(data_directory, clean=False):


    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    CSV_DIR = DATA_DIR / "csv"


    clean_status = "clean" if clean else "raw"
    data_output_path = CSV_DIR / f"20news-{clean_status}.csv"
    label_output_path = CSV_DIR / f"20news-label.csv"


    rows = []

    for split in os.listdir(data_directory):
        split_dir = os.path.join(data_directory, split)

        if not os.path.isdir(split_dir): continue

        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)

            if not os.path.isdir(label_dir): continue

            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)

                with open(file_path, "r", encoding="latin1") as f:
                    content = f.read()

                    rows.append({
                        "id": filename,
                        "label": label,
                        "content_raw": content,
                    })

    df = pd.DataFrame(rows)
    print(f"Preprocessing {data_directory} to bulk csv at {data_output_path}")
    print(f"\tCleaning = {clean}")
    print(f"\tShape {df.shape} for {df.columns}")
    
    print("\tGenerating Data CSV")
    df.to_csv(data_output_path, index=False)

    hier_df = df[['id', 'label']].copy()
    hier_df[['superlabel', 'sublabel']] = hier_df['label'].apply(lambda label: pd.Series(split_label_hierarchy(label)))
    
    print("\tGenerating Label CSV")
    hier_df.to_csv(label_output_path, index=False)

    print("\tCompleted preprocessing.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert a folder of 20 Newsgroups data to csv")
    parser.add_argument("-d", "--data", type=str, help="Path to top-level directory of 20newsgroups containing test/train subfolders")
    parser.add_argument("-c", "--clean", action="store_true", default=False, help="Toggle cleaning on or off, default is off")
    args = parser.parse_args()

    folder_to_csv(data_directory=args.data, clean=args.clean)