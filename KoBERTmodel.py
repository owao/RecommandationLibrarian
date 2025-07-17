import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

class KoBERTEmbedder:
    def __init__(self):
        #self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        #self.model = BertModel.from_pretrained("monologg/kobert")
        #self.tokenizer = BertTokenizer.from_pretrained("skt/kobert-base-v1", do_lower_case=False)
        #self.model = BertModel.from_pretrained("skt/kobert-base-v1")
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
        self.model = AutoModel.from_pretrained("skt/kobert-base-v1")

        self.model.eval()

    def text_to_vector(self, text: str) -> np.ndarray:
        """
        입력 텍스트를 KoBERT 임베딩 벡터로 변환
        """
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        cls_vector = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
        return cls_vector

    def find_closest_rdf2vec(self, vector: np.ndarray, rdf2vec_model) -> np.ndarray:
        """
        KoBERT 벡터에 가장 유사한 RDF2Vec 벡터를 반환 (단어가 아닌 벡터값)
        """
        best_score = -1
        best_vector = None

        for word in rdf2vec_model.wv.index_to_key:
            rdf_vec = rdf2vec_model.wv[word]
            sim = cosine_similarity([vector], [rdf_vec])[0][0]
            if sim > best_score:
                best_score = sim
                best_vector = rdf_vec

        return best_vector

    def vector_to_text(self, vector: np.ndarray, rdf2vec_model) -> str:
        """
        주어진 벡터와 가장 유사한 RDF2Vec 단어를 반환
        """
        best_score = -1
        best_word = None

        for word in rdf2vec_model.wv.index_to_key:
            rdf_vec = rdf2vec_model.wv[word]
            sim = cosine_similarity([vector], [rdf_vec])[0][0]
            if sim > best_score:
                best_score = sim
                best_word = word

        return best_word