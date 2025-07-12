from tqdm import tqdm
import pandas as pd
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
from collections import defaultdict


# --- 그래프와 엔티티 로드 ---
def load_graph(file_path):
    df = pd.read_csv(file_path, sep="\t")
    triples = [tuple(x) for x in df.to_numpy()]

    graph = KG()
    for s, p, o in triples:
        subj = Vertex(s)
        obj = Vertex(o)
        pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
        graph.add_walk(subj, pred, obj)

    entities = list(set(s for (s, _, _) in triples) | set(o for (_, _, o) in triples))
    return graph, entities, triples


# --- 디버깅 코드는 여기 안에 ---
if __name__ == "__main__":
    file_path = "traindataset/wikidata_book_subgraph_7hop.tsv"
    graph, entities, triples = load_graph(file_path)

    print(f"📦 총 트리플 수: {len(triples)}")
    print("🧪 예시 트리플 5개:")
    for t in triples[:5]:
        print(f"   {t}")

    connections = defaultdict(int)
    for s, p, o in triples:
        connections[s] += 1
        connections[o] += 1

    print(f"🌐 그래프에 추가된 주체 수: {len(set(s for s, _, _ in triples))}")
    print(f"🔍 엔티티 수 (주어+목적어): {len(entities)}")
    print(f"📊 평균 연결 수: {sum(connections.values()) / len(connections):.2f}")
    print("🧪 연결 많은 상위 5개:")
    for uri, count in sorted(connections.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {uri}: {count}개 연결")

    # 🧪 Walk 샘플 보기 (에러 방지용)
    rdf2vec = RDF2VecTransformer(
        Word2Vec(sg=1, window=5, min_count=1),
        walkers=[RandomWalker(max_depth=4)]
    )

    print("🚶 Walk 샘플 생성:")
    for entity in tqdm(entities[:10], desc="🔍 Walk 생성 중", unit="entity"):
        try:
            entity_walks = rdf2vec.get_walks(graph, [entity])
            print(f"✅ {entity} → {len(entity_walks[0]) if entity_walks else 0}개 walk")
        except Exception as e:
            print(f"❌ {entity} walk 실패: {e}")
