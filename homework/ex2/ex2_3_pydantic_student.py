import pprint
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_config import get_llm


# 출력 구조를 정의하는 Pydantic 모델 (각 필드에 타입과 설명)
class StudentInfo(BaseModel):
    """학생 자기소개에서 추출한 정보"""
    name: str = Field(description="학생 이름")
    age: int = Field(description="학생 나이 (숫자만)")
    major: str = Field(description="전공 학과")
    hobbies: List[str] = Field(description="취미 목록 (각 취미를 하나의 항목으로)")
    goal: str = Field(description="앞으로의 목표")


# Pydantic 출력 파서 초기화
parser = PydanticOutputParser(pydantic_object=StudentInfo)
format_instructions = parser.get_format_instructions()

# ChatPromptTemplate (부분 변수 미리 설정)
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 자기소개 글에서 학생 정보를 추출하는 전문가입니다.
다음 형식을 **정확히** 따르세요:
{format_instructions}"""),
    ("user", "자기소개: {intro}"),
]).partial(format_instructions=format_instructions)

# LLM 초기화 (정보 추출은 정확성이 중요하므로 temperature 를 낮추고 seed 고정)
model = get_llm(temperature=0.1, seed=42)

# 체인 생성 (LCEL)
chain = prompt | model | parser

# 자유 형식 자기소개 입력
intro = ("안녕하세요! 저는 김민수이고 22살입니다. 컴퓨터공학을 전공하고 있어요. "
         "취미로는 게임하기, 영화보기, 코딩을 좋아합니다. 앞으로 훌륭한 개발자가 되는 것이 목표입니다.")

# 실행
try:
    output = chain.invoke({"intro": intro})

    print("\n===== 추출 결과 =====")
    print(type(output))
    pprint.pprint(output.model_dump(), width=80, indent=2)  # Pydantic 객체를 dict로 출력

    # 필드별 접근: 객체라서 점(.)으로 꺼냄
    print(f"\n이름: {output.name}")
    print(f"나이: {output.age} ({type(output.age).__name__})")
    print(f"취미: {', '.join(output.hobbies)}")  # List => str 로 변환
except Exception as e:
    print(f" 오류: {e}")