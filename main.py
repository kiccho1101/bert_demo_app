# %%
import torch
import string

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForMaskedLM
import umap
import sa_train

print("loading BertTokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("loading BertForMaskedLM...")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").eval()
print("loading SentenceTransformer...")
bert_st = SentenceTransformer("bert-base-nli-mean-tokens")
print("loading Sentiment Analysis Model (this may take a long time)...")
sa = sa_train.train_sa_model()

top_k = 10


def vectorize_sentences(text_sentence: List[str]) -> Dict[str, List[np.array]]:
    return {"bert": [nparray.tolist() for nparray in bert_st.encode(text_sentence)]}


def sentiment_analysis(text_sentence: str) -> float:
    text_vectorized = vectorize_sentences([text_sentence])["bert"]
    score = sa.predict_proba(text_vectorized)[0, 1]
    return score


def umap_comp(
    data: List[List[float]],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> List[List[float]]:
    return (
        umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        .fit_transform(data)
        .tolist()
    )


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + "[PAD]"
    tokens = []
    for w in pred_idx:
        token = "".join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace("##", ""))
    return "\n".join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace("<mask>", tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += " ."

    input_ids = torch.tensor(
        [tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)]
    )
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(
        bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean
    )

    return {"bert": bert}
