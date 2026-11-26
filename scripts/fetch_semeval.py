"""Download the REAL SemEval-2014 laptop-review ABSA data (via the Hugging Face copy) into data/semeval_laptops.csv.
Not committed to the repo: check the dataset's terms before redistributing it."""
import csv, io, os, urllib.request
import pandas as pd  # pip install pandas pyarrow

BASE = "https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-laptops/resolve/main/data/{}-00000-of-00001.parquet"
dfs = [pd.read_parquet(io.BytesIO(urllib.request.urlopen(BASE.format(s)).read())) for s in ("train", "test")]
df = pd.concat(dfs)
os.makedirs("data", exist_ok=True)
texts = list(dict.fromkeys(df["text"]))
spans = df.groupby("text")["span"].apply(lambda s: "|".join(sorted(set(s)))).to_dict()
with open("data/semeval_laptops.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "rating", "text", "aspects"])
    for i, t in enumerate(texts, 1):
        w.writerow([f"L{i}", 0, t, spans[t]])
print(len(texts), "unique review sentences -> data/semeval_laptops.csv")
