from enum import Enum
from langchain_classic.output_parsers import EnumOutputParser  # langchain 1.x: classic 패키지에서 import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from llm_config import get_llm


# 허용할 분류 값을 Enum 으로 정의 (이름 = 코드에서 쓰는 값, 값 = 모델이 답해야 하는 문자열)
class Sentiment(Enum):
    """영화 리뷰 감정 분류 값 (과제 지문: 긍정 / 부정 / 보통)"""
    POSITIVE = "긍정"
    NEGATIVE = "부정"
    NEUTRAL = "보통"


# Enum 출력 파서 초기화
parser = EnumOutputParser(enum=Sentiment)

# 출력 형식 지침 확인: "Select one of the following options: 긍정, 부정, 보통"
format_instructions = parser.get_format_instructions()
print(format_instructions)

# 프롬프트 템플릿 설정 (형식 지침을 미리 주입)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 영화 리뷰 감정 분류 전문가입니다. 리뷰를 읽고 감정을 분류하세요. 설명 없이 값만 답하세요.\n{format_instructions}"),
    ("user", "리뷰: {review}"),
]).partial(format_instructions=format_instructions)

# LLM 초기화 (분류 작업이므로 temperature 를 0으로 설정)
llm = get_llm(temperature=0)

# 체인 생성: 프롬프트 → 모델 → Enum 파서
chain = prompt | llm | parser

# 여러 개의 테스트 리뷰로 검증
reviews = [
    "이 영화 정말 재미없어요. 시간 낭비였습니다.",
    "배우들의 연기가 훌륭하고 스토리도 감동적이었어요!",
    "그냥 무난한 영화였습니다. 나쁘지도 좋지도 않아요.",
]

for review in reviews:
    print(f"리뷰: {review}")
    try:
        result = chain.invoke({"review": review})
        # 결과는 문자열이 아니라 Sentiment Enum 멤버
        print(f"  분류 결과: {result} / 타입: {type(result).__name__} / 값: {result.value}")
    except OutputParserException as e:
        # 모델이 Enum 에 없는 값으로 답하면 파싱 오류가 발생
        print(f"  파싱 오류: {e}")
    print("-" * 50)