import os
import requests
from dotenv import load_dotenv
from pprint import pprint

# OpenRouter API 설정
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("OPENROUTER_API_URL")

# 핵심 함수: 문장에서 키워드와 SPARQL 논리 쿼리 추출
def extract_keywords_sparql(sentence: str):
    prompt = f"""
문장: "{sentence}"

문장에서 **명사(Noun)만 추출**해서 다음 두 그룹으로 분류해줘:

- 긍정적으로 언급된 명사 (예: 맛있다고 표현된 과일)
- 부정적으로 언급된 명사 (예: 맛이 없다고 표현된 채소)

결과는 다음 형식으로 출력해줘:

키워드 리스트: ["명사1", "명사2", "명사3" ... ]
SPARQL 쿼리:
FILTER ((?label = "긍정명사1" || ?label = "긍정명사2" ...) && !(?label = "부정명사1" || ?label = "부정명사2" ...))

키워드 리스트는 긍정 명사를 나열하고 부정 명사를 나열하는 순으로 써줘.

    """

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [
            {"role": "system", "content": "당신은 텍스트에서 명사 키워드와 논리 쿼리를 추출하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"❌ 오류 발생: {response.status_code} - {response.text}"
    
# 타입 변환을 위한 파싱 함수
def parse_response(text: str):
    try:
        keyword_part = text.split("키워드 리스트:")[1].split("SPARQL 쿼리:")[0].strip()
        sparql_part = text.split("SPARQL 쿼리:")[1].strip()

        # 리스트 문자열을 실제 파이썬 리스트로 변환
        keywords = eval(keyword_part)  # ⚠️ 이건 LLM이 정확히 리스트 형식으로 줄 때만 안전

        return keywords, sparql_part
    except Exception as e:
        print("❌ 파싱 실패:", e)
        return [], ""

# 단순 테스트용
if __name__ == "__main__":
    first_result = extract_keywords_sparql("어제 사과랑 바나나, 오이를 먹었는데 오이는 맛없었지만 나머지는 달고 맛있었어. 수박도 먹었는데 그것도 맛이 없었어.")
    result1, result2 = parse_response(first_result)
    print(result1)
    print(result2)
