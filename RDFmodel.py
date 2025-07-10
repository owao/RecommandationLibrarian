import pandas as pd
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
from collections import defaultdict

# 파일에서 트리플 로드
df = pd.read_csv("traindataset/wikidata_book_subgraph_7hop.tsv", sep="\t")
triples = [tuple(x) for x in df.to_numpy()]

# Knowledge Graph 생성
graph = KG()
for s, p, o in triples:
    subj = Vertex(s)
    obj = Vertex(o)
    pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
    graph.add_walk(subj, pred, obj)


# 모든 엔티티 수집 (주어 + 목적어)
entities = list(set(s for (s, _, _) in triples) | set(o for (_, _, o) in triples))

# 임베딩 모델 정의
walker = RandomWalker(max_depth=10)
rdf2vec = RDF2VecTransformer(Word2Vec(sg=1, window=5, min_count=1), walkers=[walker])

# 엔티티당 연결 수 확인
connections = defaultdict(int)
for s, p, o in triples:
    connections[s] += 1
    connections[o] += 1

print(f"평균 연결 수: {sum(connections.values()) / len(connections):.2f}")


if __name__ == "__main__":
    print("📚 RDF2Vec 임베딩 학습 중...")
    embeddings = rdf2vec.fit_transform(graph, entities)
    print("✅ 임베딩 완료!")

    print("총 임베딩 수:", len(embeddings))
    print("하나당 벡터 차원:", len(embeddings[0]))
    

