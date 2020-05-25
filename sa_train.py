# %%
import re
import urllib
import pandas as pd
import tarfile
import os
from os.path import isfile, join
from os import system, listdir
from random import shuffle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import pickle


def download_Imdb_data():
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "./data/aclImdb_v1.tar.gz"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    tar = tarfile.open("./data/aclImdb_v1.tar.gz")
    tar.extractall("./data/")
    tar.close()


def create_data_frame(folder: str) -> pd.DataFrame:
    """
    folder - the root folder of train or test dataset
    Returns: a DataFrame with the combined data from the input folder
    """
    pos_folder = f"{folder}/pos"  # positive reviews
    neg_folder = f"{folder}/neg"  # negative reviews

    def get_files(fld: str) -> list:
        """
        fld - positive or negative reviews folder
        Returns: a list with all files in input folder
        """
        return [join(fld, f) for f in listdir(fld) if isfile(join(fld, f))]

    def append_files_data(data_list: list, files: list, label: int) -> None:
        """
        Appends to 'data_list' tuples of form (file content, label)
        for each file in 'files' input list
        """
        for file_path in files:
            with open(file_path, "r") as f:
                text = f.read()
                data_list.append((text, label))

    pos_files = get_files(pos_folder)
    neg_files = get_files(neg_folder)

    data_list = []
    append_files_data(data_list, pos_files, 1)
    append_files_data(data_list, neg_files, 0)
    shuffle(data_list)

    text, label = tuple(zip(*data_list))
    # replacing line breaks with spaces
    text = list(map(lambda txt: re.sub("(<br\s*/?>)+", " ", txt), text))

    return pd.DataFrame({"text": text, "label": label})


def train_sa_model():
    if not os.path.exists("sa_model.pkl"):
        download_Imdb_data()
        df = create_data_frame("./data/aclImdb/train")
        lr = LogisticRegression()
        bert_st = SentenceTransformer("bert-base-nli-mean-tokens")
        texts_vectorized = bert_st.encode(df["text"].tolist())
        lr.fit(texts_vectorized, df["label"])
        pickle.dump(lr, open("sa_model.pkl", "wb"))
    else:
        lr = pickle.load(open("sa_model.pkl", "rb"))
    return lr
