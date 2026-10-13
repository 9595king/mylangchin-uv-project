import json
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema  # langchain 1.x: classic 패키지에서 import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from llm_config import get_llm

# 출력할 필드를 ResponseSchema 로 정의 (name: 결과 dict 의 키, description: 모델에게 알려줄 설명)
response_schemas = [
    ResponseSchema(name="destination", description="여행지 (도시 또는 지역 이름)"),
    ResponseSchema(name="duration", description="여행 기간 (예: 2박 3일)"),
    ResponseSchema(name="budget", description="총 여행 예산 또는 사용 금액 (예: 30만원)"),
    ResponseSchema(name="rating", description="여행 추천도, 1~5 사이 숫자 하나"),
    ResponseSchema(name="activities", description="여행 중 한 주요 활동 목록", type="List[string]"),
]

# ResponseSchema 목록으로 파서 생성
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 출력 형식 지침 확인: ```json 코드 블록과 필드별 설명이 포함됨
format_instructions = parser.get_format_instructions()
print(format_instructions)

# 프롬프트 템플릿 설정 (형식 지침을 미리 주입)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 여행 후기 분석 전문가입니다. 여행 후기에서 핵심 정보를 추출하고 반드시 아래 형식을 지키세요.\n{format_instructions}"),
    ("user", "여행 후기: {text}"),
]).partial(format_instructions=format_instructions)

# LLM 초기화 (구조화 출력은 일관성이 중요하므로 temperature 를 0으로 설정)
llm = get_llm(temperature=0)

# 체인 생성: 프롬프트 → 모델 → 구조화 파서
chain = prompt | llm | parser

text = ("지난 주에 부산으로 2박 3일 여행을 다녀왔어요. 총 30만원 정도 썼는데 해운대에서 바다구경하고, "
        "자갈치시장에서 회 먹고, 감천문화마을도 구경했어요. 정말 만족스러운 여행이었습니다. "
        "5점 만점에 4점 정도 줄 수 있을 것 같아요.")

try:
    response = chain.invoke({"text": text})
    print(type(response))  # 결과는 dict

    # JSON 데이터 출력
    print(json.dumps(response, indent=4, ensure_ascii=False))

    # 필드별 접근: ResponseSchema 의 name 이 dict 의 키
    print("여행지:", response["destination"])
    print("주요 활동:", ", ".join(response["activities"]))
except OutputParserException as e:
    # 모델이 형식을 지키지 않았거나 필드가 빠진 경우
    print(f"파싱 오류: {str(e).splitlines()[0]}")