import requests, urllib.parse, time
from bs4 import BeautifulSoup
from tqdm import tqdm              # 진행률 표시
from pprint import pprint

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

def crawl_aladin(keyword: str, top_n: int = 5) -> list[dict]:
    """
    알라딘 검색 결과에서 상위 `top_n`권의
    제목·저자·장르(카테고리)·출판사를 추출해 반환한다.
    """
    base = "https://www.aladin.co.kr/search/wsearchresult.aspx"
    params = {"SearchTarget": "All", "SearchWord": keyword}
    url = f"{base}?{urllib.parse.urlencode(params, safe=':')}"
    res = requests.get(url, headers=HEADERS, timeout=10)
    res.raise_for_status()              # 200이 아니면 예외 발생

    soup = BeautifulSoup(res.text, "html.parser")

    books = []
    # └ div.ss_book_list 하나가 책 1권에 대응
    i = 1
    for box in soup.select("div.ss_book_list")[:top_n]:
        if i%2 == 0:
            i += 1
            continue
        # ① 제목
        title_tag = box.select_one("a.bo3")
        title = title_tag.get_text(strip=True) if title_tag else "N/A"

        # ② 저자·출판사 (형식: “저자 | 출판사 | YYYY.MM.DD”)
        author, publisher = "N/A", "N/A"
        for li in box.select("ul li"):
            if " | " in li.get_text():
                parts = [p.strip() for p in li.get_text().split("|")]
                if len(parts) >= 2:
                    author, publisher = parts[0], parts[1]
                break
        books.append(
            {"keyword": keyword, "title": title,
             "author": author, "publisher": publisher}
        )
        i += 1
    return books


def search_wordlist(keywords: list, top_n: int = 5) -> list[dict]:
    """"
    여러 단어에 대해 crawl_aladin을 수행하고 결과 리스트를 반환한다.
    """
    all_results = []
    for kw in tqdm(keywords, desc="Aladin crawl"):
        all_results.extend(crawl_aladin(kw, top_n*2))
        time.sleep(1)
    return all_results

# 단순 테스트용
if __name__ == "__main__":
    keywords = ["파이썬", "인공지능", "주식 투자"]   # ★ 원하는 키워드 목록
    all_results = []
    for kw in tqdm(keywords, desc="Aladin crawl"):
        all_results.extend(crawl_aladin(kw))
        time.sleep(1)                 # 딜레이 1초 (IP 안전장치)

    pprint(all_results)               # 확인용
