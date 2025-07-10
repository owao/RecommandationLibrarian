from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time

# SPARQL 설정
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)
sparql.setTimeout(60)

# 시작 엔티티: 책 (Q571)
start_entity = "http://www.wikidata.org/entity/Q571"
visited = set()
triples = []

# 의미 있는 predicate만 필터링 (장르, 주제, 상위 개념, 인스턴스, 구성 요소 등)
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
    LIMIT 30
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
for hop in range(3):
    print(f"🚀 Hop {hop + 1} 시작")
    next_frontier = []
    for entity in frontier:
        if entity in visited:
            continue
        visited.add(entity)
        edges = get_one_hop(entity)
        triples.extend(edges)
        for (_, _, obj) in edges:
            if obj not in visited:
                next_frontier.append(obj)
        time.sleep(3)  # 서버 과부하 방지
    frontier = next_frontier

# 결과 저장
df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
df.to_csv("traindataset/wikidata_book_subgraph_3hop.tsv", sep="\t", index=False)
print("✅ 3-hop 서브그래프 저장 완료")
