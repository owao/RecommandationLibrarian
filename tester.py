import book_crawler
import extractwords
from pprint import pprint

if __name__ == "__main__":
    
    sentence = input("추천받고 싶은 내용을 적어주세요.")
    extraction = extractwords.extract_keywords_sparql(sentence)
    keywords, query = extractwords.parse_response(extraction)
    ###### N/A 있으면 걸러내기
    ###### 부정 단어 빼기

    top_n = int(input("몇 권을 추천받으시겠습니까?"))
    #### 멘트 바꿔라

    all_results = book_crawler.search_wordlist(keywords, top_n)

    pprint(all_results)
