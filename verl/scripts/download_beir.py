
from beir import util
import os

dataset = "fiqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

out_dir = "datasets"
data_path = os.path.join(out_dir, dataset)

if not os.path.exists(data_path):
    util.download_and_unzip(url, out_dir)

print("Dataset downloaded to:", data_path)
