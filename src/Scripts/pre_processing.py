import torch
from sentence_transformers import SentenceTransformer, models
#dataset_in=np.load('../../data/cleaned2.npy', allow_pickle=True) en gros
#path_out='../../data/multilingual_embeddings.npy'
def embedder(dataset_in, path_out, save):
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    embeddings= model.encode(dataset_in, batch_size=8, show_progress_bar=True)
    if save=True:
        np.save(paht_out, embeddings, allow_pickle=True)
    return(embeddings)


