import pandas as pd
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
import os

# KG 클래스에 hash 가능하도록 패치
def kg_hash(self):
    return id(self)
KG.__hash__ = kg_hash

def flatten_walks(walks):
    clean_walks = []

    for walk in walks:
        flat = []
        for triple in walk:
            if isinstance(triple, tuple):  # (subj, pred, obj)
                for node in triple:
                    name = node.name if hasattr(node, "name") else str(node)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="ignore")
                    flat.append(str(name))
            else:
                name = triple.name if hasattr(triple, "name") else str(triple)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="ignore")
                flat.append(str(name))
        clean_walks.append(flat)
    return clean_walks


if __name__ == "__main__":
    # 1. Triple 로딩
    df = pd.read_csv("traindataset/wikidata_book_subgraph_3hop.tsv", sep="\t")
    df = df.iloc[1:]
    triples = [tuple(x) for x in df.to_numpy()]

    # 2. Knowledge Graph 생성
    graph = KG()
    for s, p, o in triples:
        subj = Vertex(str(s))
        obj = Vertex(str(o))
        pred = Vertex(str(p), predicate=True, vprev=subj, vnext=obj)
        graph.add_walk(subj, pred, obj)

    # 3. 엔티티 추출
    entities = list(set(s for (s, _, _) in triples) | set(o for (_, _, o) in triples))

    # 4. RDF2Vec 설정
    walker = RandomWalker(max_depth=5)
    embedder = Word2Vec(sg=1, window=5, min_count=1)
    rdf2vec = RDF2VecTransformer(embedder, walkers=[walker])

    # 5. Walk 생성
    print("📚 Walk 생성 중...")
    walks = rdf2vec.get_walks(graph, entities)
    flattened_walks = [
        [str(v) for v in walk]
        for walk in walks
    ]
    
    # ✅ 문자열 변환
    clean_walks = []
    for walk in walks:
        clean_walks.append([str(token) for token in walk])
    clean_walks = flatten_walks(walks)


    print(f"총 walk 수: {len(clean_walks)}")
    # walk 내부 형태 확인
    print("🔍 walk 샘플 보기")
    for i, walk in enumerate(walks[:5]):
        print(f"Walk {i}: {[type(token) for token in walk]}")
        print(f"Raw: {walk}")


    # 6. 학습
    print("🧠 Word2Vec 학습 중...")
    rdf2vec.fit(clean_walks, entities)

    # 7. 임베딩
    print("🎯 임베딩 추출 중...")
    embeddings = rdf2vec.transform(graph, entities)

    # 8. 저장
    output_path = "model/rdf2vec_book_3hop.model"
    os.makedirs("model", exist_ok=True)
    embedder._model.save(output_path)
    print("✅ 모델 저장 완료:", output_path)

    # 9. 임베딩 확인
    print("✅ 임베딩 완료!")
    print("총 임베딩 수:", len(embeddings))
    if embeddings:
        print("하나당 벡터 차원:", len(embeddings[0]))

    print("Word2Vec 학습된 단어 수:", len(embedder._model.wv))
    print("예시 단어들:", list(embedder._model.wv.index_to_key[:5]))


'''
import pandas as pd
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.graphs import Vertex

# 1. triple 파일 불러오기
df = pd.read_csv("traindataset/wikidata_all_subgraph_3hop.tsv", sep="\t")
triples = [tuple(x) for x in df.to_numpy()]

# 2. Knowledge Graph 생성
graph = KG()
for s, p, o in triples:
    graph.add_triple(s, p, o)  # pyRDF2Vec 0.2.3+ 에서 공식 지원되는 방식

# 3. 엔티티 URI 리스트 추출
entities = list(set(s for (s, _, _) in triples) | set(o for (_, _, o) in triples))

# 4. RDF2VecTransformer 구성
transformer = RDF2VecTransformer(
    embedder=Word2Vec(vector_size=100, window=5, sg=1, min_count=1, workers=1),
    walkers=[RandomWalker(depth=4, n_walks=10, with_reverse=True, n_jobs=2)],
    verbose=1
)

# 5. 학습 및 임베딩 생성
print("🚀 fit_transform 시작...")
embeddings, _ = transformer.fit_transform(graph, entities)

# 6. 결과 확인
print("✅ 총 임베딩 개수:", len(embeddings))
print("✅ 하나당 벡터 차원:", len(embeddings[0]) if embeddings else "N/A")
'''