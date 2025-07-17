import book_crawler
import extractwords
import KoBERTmodel
from pprint import pprint

from KoBERTmodel import KoBERTEmbedder
from gensim.models import Word2Vec

if __name__ == "__main__":

    rdf2vec_model = Word2Vec.load("model/rdf2vec_all_3hop.model")

    #1. 입력을 받는다 
    sentence = input("추천받고 싶은 내용을 적어주세요.")
    top_n = int(input("몇 권을 추천받으시겠습니까?"))

    #2. A단계 - 단어 추출
    extraction = extractwords.extract_keywords_sparql(sentence)
    keywords, query = extractwords.parse_response(extraction)

    #3. B단계 - KoBERT에 단어를 넣어서 연관 단어를 뽑기
    embedder = KoBERTEmbedder()
    answer_keywords = []
    for i in keywords:
        vector = embedder.text_to_vector(i)
        return_vector = embedder.find_closest_rdf2vec(vector, rdf2vec_model)
        answer_keywords.append(embedder.vector_to_text(vector, rdf2vec_model))

    #4. C단계 - 크롤링 후 책 출력
    all_results = book_crawler.search_wordlist(answer_keywords, top_n)

    pprint(all_results)
