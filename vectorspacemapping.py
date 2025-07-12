import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------------------
# 1. 저장된 RDF2Vec 모델 로드
# ---------------------------
rdf2vec_model_path = "models/rdf2vec_model.model"  # 너가 저장한 경로로 수정해
rdf2vec_model = Word2Vec.load(rdf2vec_model_path)

print("✅ RDF2Vec 모델 로드 완료!")
print(f"🧠 단어 수: {len(rdf2vec_model.wv.index_to_key)}")
print(f"📐 임베딩 차원: {rdf2vec_model.vector_size}")

# ---------------------------
# 2. KoBERT 임베딩 생성
# ---------------------------
kobert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")  # HuggingFace에서 KoBERT 기반 SBERT

# 테스트할 label 목록 (예: Wikidata entity label들)
labels = [
    "book", "writer", "detective novel", "fantasy", "publishing house",
    "essay", "romance", "non-fiction", "autobiography", "light novel"
]

kobert_embeddings = kobert.encode(labels, convert_to_numpy=True)

# ---------------------------
# 3. RDF2Vec 단어 벡터 추출 (label 이름과 매칭되는 경우만)
# ---------------------------
rdf_embeddings = []
valid_labels = []
for label in labels:
    if label in rdf2vec_model.wv:
        rdf_embeddings.append(rdf2vec_model.wv[label])
        valid_labels.append(label)

rdf_embeddings = np.array(rdf_embeddings)
print(f"🎯 RDF2Vec에 존재하는 라벨 수: {len(valid_labels)}")

# ---------------------------
# 4. FAISS 기반 유사도 검색 매핑
# ---------------------------
dim = rdf2vec_model.vector_size
index = faiss.IndexFlatL2(dim)
index.add(rdf_embeddings)

print("🔎 KoBERT 임베딩 → RDF2Vec 임베딩 유사한 것 찾기")
for i, label in enumerate(labels):
    if label not in valid_labels:
        print(f"❌ '{label}'은 RDF2Vec에 없음. 건너뜀.")
        continue
    vec = kobert_embeddings[i].astype('float32')
    D, I = index.search(np.array([vec]), k=3)
    print(f"\n💬 '{label}'와 유사한 RDF2Vec 벡터:")
    for idx, dist in zip(I[0], D[0]):
        print(f"   - {valid_labels[idx]} (거리: {dist:.4f})")
