import os
import re
import nltk
import pandas as pd

from pathlib import Path
import argparse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = None
lemmatizer = None

def nltk_resources():
    global stop_words, lemmatizer

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

def clean_text(text):

    parts = text.split('\n\n', 1)
    
    text = parts[1] if len(parts) > 1 else parts[0]

    # Step 2: Remove signature blocks (footers) - lines after "-- " marker
    text = re.sub(r'\n--\s*\n.*', '', text, flags=re.DOTALL)

    # Step 3: Remove quoted lines (lines starting with > or |)
    lines = text.split('\n')
    lines = [line for line in lines if not line.strip().startswith('>') and not line.strip().startswith('|')]
    text = '\n'.join(lines)

    # Step 4: Remove remaining metadata patterns and quoted replies
    text = re.sub(r'(writes in message|wrote:|writes:).*', ' ', text, flags=re.IGNORECASE)

    # Step 5: Lowercase
    text = text.lower()

    # Step 6: Remove URLs, numbers, punctuation, etc.
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", " ", text)

    # Step 7: Tokenize
    tokens = text.split()

    # Step 8: Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Step 9: Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def split_label_hierarchy(label):
    parts = label.split(".")
    superlabel = parts[0] if parts else label
    sublabel = ".".join(parts[:2]) if len(parts) >= 2 else label
    return superlabel, sublabel


def folder_to_csv(data_directory):

    nltk_resources()

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"
    CSV_DIR = DATA_DIR / "csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    raw_output_path = CSV_DIR / "20news-raw.csv"
    clean_output_path = CSV_DIR / f"20news-clean.csv"
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
                
                cleaned = clean_text(content)

                rows.append({
                    "id": filename,
                    "label": label,
                    "content_clean": cleaned,
                    "content_raw": content,
                })

    df = pd.DataFrame(rows)
    df[["id", "label", "content_raw"]].to_csv(raw_output_path, index=False)
    df[["id", "label", "content_clean"]].to_csv(clean_output_path, index=False)

    hier_df = df[['id', 'label']].copy()
    hier_df[['superlabel', 'sublabel']] = hier_df['label'].apply(lambda label: pd.Series(split_label_hierarchy(label)))
    
    print("\tGenerating Label CSV")
    hier_df.to_csv(label_output_path, index=False)

    print("\tCompleted preprocessing.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert a folder of 20 Newsgroups data to csv")
    parser.add_argument("-d", "--data", type=str, help="Path to top-level directory of 20newsgroups containing test/train subfolders")
    args = parser.parse_args()

    folder_to_csv(data_directory=args.data)