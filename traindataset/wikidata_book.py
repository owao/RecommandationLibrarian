from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time
from tqdm import tqdm  # ✅ tqdm 추가

N = int(input("k-hop의 단계 수를 입력하세요"))
#THEME = input("중심 주제어를 입력하세요")

# SPARQL 설정
sparql = SPARQLWrapper("https://query.wikidata.org/sparql") 
sparql.setReturnFormat(JSON)
sparql.setTimeout(60)

# 시작 엔티티: 책 (Q571)
start_entity = "http://www.wikidata.org/entity/Q571"
print(start_entity)
visited = set()
triples = []

def get_one_hop(entity):
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?p ?o WHERE {{
        <{entity}> ?p ?o .
        FILTER(?p IN (
            wdt:P136,  # 장르
            wdt:P921,  # 주제
            wdt:P279,  # 하위 개념
            wdt:P31,   # 인스턴스
            wdt:P527   # 구성 요소
        ))
        FILTER isIRI(?o)
    }}
    LIMIT 50
    """
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return [(entity, r["p"]["value"], r["o"]["value"]) for r in results["results"]["bindings"]]
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return []

# hop-by-hop 탐색
frontier = [start_entity]
for hop in range(N):
    print(f"🚀 Hop {hop + 1} 시작")
    next_frontier = []
    for entity in tqdm(frontier, desc=f"➡️ Hop {hop + 1} 진행 중", unit="entity"):
        if entity in visited:
            continue
        visited.add(entity)
        edges = get_one_hop(entity)
        triples.extend(edges)
        for (_, _, obj) in edges:
            if obj not in visited:
                next_frontier.append(obj)
        time.sleep(2)
    frontier = next_frontier

# 결과 저장
df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
df.to_csv(f"traindataset/wikidata_book_subgraph_{N}hop.tsv", sep="\t", index=False)
print(f"✅ {N}-hop 서브그래프 저장 완료")
