from SPARQLWrapper import SPARQLWrapper, JSON
from deep_translator import GoogleTranslator
import time

# 영어 라벨 리스트 예시
english_labels = ["apple", "banana", "melon"]

# SPARQL 설정
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

# RDF2Vec 입력 저장용
rdf2vec_triples = []

# 결과 출력용
results_list = []

for label in english_labels:
    query = f"""
    SELECT ?item ?itemLabel ?itemLabel_ko WHERE {{
      ?item rdfs:label "{label}"@en.
      OPTIONAL {{
        ?item rdfs:label ?itemLabel_ko.
        FILTER (lang(?itemLabel_ko) = "ko")
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    time.sleep(1)
    results = sparql.query().convert()
    bindings = results['results']['bindings']

    if not bindings:
        continue

    for result in bindings:
        uri = result['item']['value']
        eng_label = result.get('itemLabel', {}).get('value', label)
        kor_label = result.get('itemLabel_ko', {}).get('value')

        # 한국어 라벨 없을 경우 번역
        if not kor_label:
            kor_label = GoogleTranslator(source='en', target='ko').translate(eng_label)

        print(f"✅ 영어: {eng_label} | 🇰🇷 한국어: {kor_label} | URI: {uri}")

        # RDF2Vec용 트리플 포맷 (subject, predicate, object)
        triple = f"<{uri}> <rdfs:label> \"{kor_label}\"@ko ."
        rdf2vec_triples.append(triple)
        results_list.append((eng_label, kor_label, uri))

# 결과 확인
print("\n📦 RDF2Vec 입력용 트리플들:")
for t in rdf2vec_triples:
    print(t)
