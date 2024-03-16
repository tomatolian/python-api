from fastapi import FastAPI
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
with open("embeddings.pkl", "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data["sentences"]
    stored_embeddings = stored_data["embeddings"]
app = FastAPI()

@app.get("/")
async def root(q= None):
    if q:
        search_word=model.encode(q)
    else:
        search_word="query not found"
    prob = np.dot(stored_embeddings, search_word)  # ベクトル間の類似度を計算
    rank = np.argsort(prob)[::-1][0 : 5]
    res=[stored_sentences[i].split("@")[0] for i in rank] 
    return {"message": "\n".join(res)}