import json
import numpy as np
import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load multilingual model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

df = pd.read_csv("/home/janek/Downloads/lena_jobs.csv", names=["job_title"])
df["embs"] = df["job_title"].apply(lambda x: model.encode(x).tolist())
df.to_csv("lena_embs.csv")

# TEST
embs = pd.read_csv("lena_embs.csv")
df["embs"] = df["embs"].apply(lambda x: np.array(x))
embedding_array = np.stack(df["embs"].values)
print(embedding_array.shape)
for job in ["Asystentka Zarządu", "Junior Analyst", "Dostawca Pierogów", "Młodszy Analityk ds. Raportowania"]:
    sims = cosine_similarity(embedding_array, model.encode([job])).flatten()
    print(job, sims.max())
