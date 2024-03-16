from sentence_transformers import SentenceTransformer
import pickle
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog.",
]
embeddings = model.encode(sentences)
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print()
with open("embeddings.pkl", "wb") as fOut:
    pickle.dump({"sentences": sentences, "embeddings": embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
with open("embeddings.pkl", "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data["sentences"]
    stored_embeddings = stored_data["embeddings"]