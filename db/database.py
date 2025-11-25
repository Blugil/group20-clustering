import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3

def split_text(text: str):
    lines = text.splitlines()
    subject = ""
    body = text

    for i, line in enumerate(lines[:30]):  
        if line.lower().startswith("subject:"):
            subject = line.partition(":")[2].strip()
            body = "\n".join(lines[i + 1 :])
            break

    return subject, body

def create_db_file(name: str = "newsgroups.db"):

    DB_DIR = Path(__file__).resolve().parent
    DB_PATH = DB_DIR / name

    if not DB_PATH.exists():
        # create the file
        print("db file not initiated, creating now")
        file = open(DB_PATH, "w")

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            body TEXT
        )
        """
    )

    connection.commit()
    return connection 

def setup_database():
    # Handle path stuff
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    CSV_DIR = DATA_DIR / "csv"

    input_csv = CSV_DIR / "20news-raw.csv"

    # Read in the CSV 
    df = pd.read_csv(input_csv)

    # Read the corpus and ids as a tolist so we don't shuffle and they are aligned
    corpus = df["content_raw"].astype(str).tolist()
    np_ids = df["id"].astype(str).to_numpy(dtype="U")


    print(np_ids[0:5])

    print("Grabbed all the ids and raw content")
    print("num ids", len(np_ids))
    print("len corpus", len(corpus))


    assert len(np_ids) == len(corpus), "len ids is not equal to len corpus"

    connection = create_db_file()
    cursor = connection.cursor()

    print("inserting rows")

    for (idx, text) in zip(np_ids, corpus):

        subject, body = split_text(text)
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents (id, subject, body)
            VALUES(?, ?, ?)
            """, (idx, subject, body)
        )
    
    connection.commit()
    connection.close()



if __name__ == "__main__":
    setup_database()